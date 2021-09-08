from typing import Callable
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch
from pathlib import Path
import os
import argparse
from ..models.multnat_model import MultNatModel
from .utils import add_generic_args


def train(
        model_cls: Callable[[argparse.Namespace], LightningModule],
        args: argparse.Namespace,
        iteration: int
):
    pl.seed_everything(args.seed)

    # init model
    model = model_cls(args)

    cdir = Path(os.path.join(
        model.hparams.checkpoint_dir, args.experiment_name,
        "version_{}".format(iteration)))
    cdir.mkdir(exist_ok=True, parents=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=cdir,
        filename=str(args.num_patterns)+'-'+str(
            args.num_tokens_per_pattern)+'-{epoch}-{val_loss:.2f}-{AUC:.2f}',
        monitor="AUC", mode="max", save_top_k=1
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        weights_summary=None,
        logger=False,
        callbacks=checkpoint_callback,
        deterministic=True,
        distributed_backend="ddp" if args.gpus > 1 else None
    )

    trainer.fit(model)

    return trainer, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser)
    MultNatModel.add_model_specific_args(parser)
    parser.add_argument('--start_num_patterns', default=100, type=int)
    parser.add_argument('--start_num_tokens_per_pattern', default=10, type=int)
    parser.add_argument('--start_version', default=0, type=int)
    args = parser.parse_args()

    out_file = "{}.tsv".format(args.experiment_name)
    if os.path.exists(out_file):
        print("Out file already exists!")
        exit(1)

    with open(out_file, 'w') as fout:
        iteration = args.start_version
        for num_patterns in [100, 75, 50, 25, 10, 5, 1]:
            if num_patterns > args.start_num_patterns:
                continue

            args.num_patterns = num_patterns
            for num_tokens_per_pattern in [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
                if num_patterns == args.start_num_patterns and\
                   num_tokens_per_pattern > args.start_num_tokens_per_pattern:
                    continue

                print('!!! {} --- {} !!!'.format(num_patterns, num_tokens_per_pattern))
                args.num_tokens_per_pattern = num_tokens_per_pattern
                trainer, model = train(MultNatModel, args, iteration)
                metrics = trainer.test(test_dataloaders=model.val_dataloader())
                val_auc = metrics[0]['AUC']
                print(num_patterns, num_tokens_per_pattern,
                      val_auc, sep='\t', file=fout)
                iteration += 1
                fout.flush()
                del trainer
                del model
                torch.cuda.empty_cache()
