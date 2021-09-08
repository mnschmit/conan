import argparse
import pytorch_lightning as pl
from ..models.multnat_model import MultNatModel
from .utils import add_dataset_specific_args, load_custom_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--classification_threshold', type=float, default=None)
    parser.add_argument('--gpus', type=int, default=1)
    add_dataset_specific_args(parser)
    args = parser.parse_args()

    model = MultNatModel.load_from_checkpoint(args.checkpoint)

    if args.classification_threshold is not None:
        model.set_classification_threshold(args.classification_threshold)

    dataloader = load_custom_data(args, model)

    trainer = pl.Trainer(gpus=args.gpus, logger=False)
    trainer.test(model, test_dataloaders=dataloader)
