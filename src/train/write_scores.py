import argparse
import pytorch_lightning as pl
from ..models.multnat_model import MultNatModel
from .utils import add_dataset_specific_args, load_custom_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('out_file')
    parser.add_argument('--gpus', type=int, default=1)
    add_dataset_specific_args(parser)
    args = parser.parse_args()

    model = MultNatModel.load_from_checkpoint(args.checkpoint)
    trainer = pl.Trainer(gpus=args.gpus, logger=False)

    dataloader = load_custom_data(args, model)
    if dataloader is None:
        model.setup(None)
        dataloader = model.val_dataloader()

    model.set_score_outfile(args.out_file)
    trainer.test(model, test_dataloaders=dataloader)
