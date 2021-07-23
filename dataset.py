import os
from torch.utils.data.dataloader import DataLoader
import random
import tarfile
import urllib.request
from itertools import groupby
from operator import itemgetter
from pathlib import Path

import gdown
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image

DATASET_DRIVE_URL = "https://drive.google.com/uc?id=10X4KdsEsqhuZBID-iMAaZMmK6BYZXIVI"
SPLIT_FILE_URL = "https://gist.githubusercontent.com/oliver-ni/d0e996193a9621c9f305396b72eb96ab/raw/8b251b2094ccba60b45bd1fdf72bbf7f5e0afa5f/train_test_val_split.txt"


def parse_datatype(item):
    for datatype in (int, float):
        try:
            return datatype(item)
        except ValueError:
            pass
    return item


def parse_and_filter(f, ids_to_load=None):
    for line in f:
        x = line.split()
        if ids_to_load is None or int(x[0]) in ids_to_load:
            yield [parse_datatype(i) for i in x]


class CUB2011(Dataset):
    def __init__(
        self,
        dataset_path,
        *,
        type=0,
        images=True,
        labels=True,
        bounding_boxes=False,
        attributes=True,
        part_locs=False,
        mturk_part_locs=False,
        transform=None,
        noise_level=0,
    ):
        root = Path(dataset_path)
        self.transform = transform

        # Get list of ids to load, based on train/test/val split
        # Reads from provided path and filters
        # 0: train
        # 1: test
        # 2: validation
        with open(root / "train_test_val_split.txt") as f:
            self.ids = [id for id, type_ in parse_and_filter(f) if type_ == type]
            ids = set(self.ids)

        if images:
            # Load image paths
            with open(root / "images.txt") as f:
                self.img_paths = {
                    id: root / "images" / fname for id, fname in parse_and_filter(f, ids)
                }

        if labels:
            # Load image labels
            with open(root / "image_class_labels.txt") as f:
                self.labels = {id: label - 1 for id, label in parse_and_filter(f, ids)}
                self.noisy = {id: random.random() < noise_level for id in self.ids}
                self.noisy_labels = {
                    id: random.randrange(200) if self.noisy[id] else self.labels[id] for id in ids
                }

        if bounding_boxes:
            # Load bounding boxes
            # <image_id> <x> <y> <width> <height>
            with open(root / "bounding_boxes.txt") as f:
                self.bounding_boxes = {
                    id: torch.tensor(box) for id, *box in parse_and_filter(f, ids)
                }

        if attributes:
            # Load binary attributes
            # Certainty and time fields excluded
            # <image_id> <attribute_id> <is_present> <certainty_id> <time>

            def fix_issue(lines):
                # There is an issue with the dataset.
                # Some lines are malformed with an extra "0"
                # See image_id #2275
                for line in lines:
                    if len(line) > 5:
                        yield line[:4] + line[5:]
                    else:
                        yield line

            with open(root / "attributes" / "image_attribute_labels.txt") as f:
                self.attributes = {
                    id: torch.tensor([float(present) for _, _, present, _, _ in fix_issue(lines)])
                    for id, lines in groupby(parse_and_filter(f, ids), key=itemgetter(0))
                }

        if part_locs:
            # Load parts
            # <image_id> <part_id> <x> <y> <visible>
            with open(root / "parts" / "part_locs.txt") as f:
                self.part_locs = {
                    id: torch.tensor([info for _, *info in lines])
                    for id, lines in groupby(parse_and_filter(f, ids), key=itemgetter(0))
                }

        if mturk_part_locs:
            # Load MTurk parts
            # Time field excluded
            # <image_id> <part_id> <x> <y> <visible> <time>
            with open(root / "parts" / "part_click_locs.txt") as f:
                self.mturk_part_locs = {
                    id: torch.tensor([info for _, *info, _ in lines])
                    for id, lines in groupby(parse_and_filter(f, ids), key=itemgetter(0))
                }

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        id = self.ids[idx]
        item = {}

        if hasattr(self, "img_paths"):
            item["img"] = read_image(str(self.img_paths[id]), mode=ImageReadMode.RGB) / 255
            if self.transform:
                item["img"] = self.transform(item["img"])

        if hasattr(self, "labels"):
            item["label"] = self.labels[id]
        if hasattr(self, "noisy"):
            item["noisy"] = self.noisy[id]
        if hasattr(self, "noisy_labels"):
            item["noisy_label"] = self.noisy_labels[id]
        if hasattr(self, "bounding_boxes"):
            item["bounding_box"] = self.bounding_boxes[id]
        if hasattr(self, "attributes"):
            item["attributes"] = self.attributes[id]
        if hasattr(self, "part_locs"):
            item["part_locs"] = self.part_locs[id]
        if hasattr(self, "mturk_part_locs"):
            item["mturk_part_locs"] = self.mturk_part_locs[id]

        return item


class CUBDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, *, noise_level=0, num_workers=0, **kwargs):
        super().__init__()

        self.dataset_path = dataset_path
        self.noise_level = noise_level
        self.num_workers = num_workers

        self.transform = {
            "train": transforms.Compose(
                [
                    transforms.Resize(160),
                    transforms.RandomRotation(45),
                    transforms.RandomResizedCrop(128),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.Resize(160),
                    transforms.CenterCrop(128),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        }

        self.dims = (1, 224, 224)

    # def prepare_data(self):
    #     if not os.path.exists("CUB_200_2011.tgz"):
    #         gdown.download(DATASET_DRIVE_URL, "CUB_200_2011.tgz")
    #     if not os.path.exists("CUB_200_2011"):
    #         tar = tarfile.open("CUB_200_2011.tgz")
    #         tar.extractall()
    #         tar.close()
    #         urllib.request.urlretrieve(SPLIT_FILE_URL, "CUB_200_2011/train_test_val_split.txt")

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.ds_train = CUB2011(
                self.dataset_path,
                type=0,
                transform=self.transform["train"],
                noise_level=self.noise_level,
            )
            self.ds_val = CUB2011(self.dataset_path, type=2, transform=self.transform["test"])

        if stage == "test" or stage is None:
            self.ds_test = CUB2011(self.dataset_path, type=1, transform=self.transform["test"])

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=64, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=64, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=64, num_workers=self.num_workers)
