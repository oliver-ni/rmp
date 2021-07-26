from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import CUBDataModule
from model import BaselineModel, BetaModel

MODELS = {"baseline": BaselineModel, "beta": BetaModel}


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", default="CUB_200_2011")
    parser.add_argument("--noise_level", type=float, default=0)
    parser.add_argument("--num_workers", type=float, default=16)

    parser.add_argument("--model", default="baseline")
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_epochs", type=float, default=20)

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    dict_args = vars(args)

    model = MODELS[dict_args["model"]](**dict_args)
    data = CUBDataModule(**dict_args)
    logger = TensorBoardLogger(
        "tensorboard",
        name=f"{dict_args['model']}_{dict_args['noise_level']}",
        default_hp_metric=False,
    )

    trainer = pl.Trainer.from_argparse_args(args, logger=logger)
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
