from dataset import CUBDataModule
from model import BaselineModel, BetaModel
import pytorch_lightning as pl
from argparse import ArgumentParser

from argparse import ArgumentParser

MODELS = {"baseline": BaselineModel, "beta": BetaModel}


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", default="CUB_200_2011")
    parser.add_argument("--noise_level", type=float, default=0)
    parser.add_argument("--num_workers", type=float, default=8)

    parser.add_argument("--model", default="baseline")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    dict_args = vars(args)

    model = BaselineModel()
    data = CUBDataModule(**dict_args)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
