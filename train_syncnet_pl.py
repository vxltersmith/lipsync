import os

import torch
from torch.utils.data import DataLoader,Subset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from Wav2LipMarlin.hparams import hparams
from SyncNetPLModel import LipSyncExpertModel
from Wav2LipMarlin.datasets.dataset import Wav2LipSyncNetDataset, FullVideoSyncDataset
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_lightning import seed_everything



# Create a TensorBoard logger
logger = TensorBoardLogger(
    save_dir='./logs/lightning_logs',   # Directory where logs will be saved
    name='syncnet_training',                    # Name of the experiment (log folder name)
)

def main():
    # Argument Parsing
    import argparse
    parser = argparse.ArgumentParser(description='Train Lip-Sync Expert Model')
    parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)
    parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
    parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', default=None, type=str)
    parser.add_argument('--accelerator', help='Device type', default="gpu", type=str)
    parser.add_argument('--devices', help='Device ids', default=1, type=int)
    parser.add_argument('--eval_every', help='Eval every N epochs', default=None, type=int)
    parser.add_argument('--num_workers', help='Device ids', default=None, type=int)
    parser.add_argument('--batch_size', help='Batch size for training', default=None, type=int)
    parser.add_argument('--limit_train_batches', help='Loop training for some batches', default=None, type=int)
    parser.add_argument('--precision', help='Loop training for some batches', default="32", type=str) # '16-mixed' for fp16
    parser.add_argument('--seed', help='Seed everything with provided value; -1 for a random seed', default=666, type=int)

    args = parser.parse_args()
    

    # Dataset and Dataloader setup
    syncnet_T = 16
    syncnet_mel_step_size = int(16 * 3.18)
    num_workers = args.num_workers if args.num_workers is not None else hparams.num_workers
    batch_size = args.batch_size if args.batch_size is not None else hparams.syncnet_batch_size
    eval_every = args.eval_every if args.eval_every is not None else 1
    limit_val_batches = 1
    
    if args.seed != -1:
        seed_everything(args.seed, workers = True) # Seed everything including workers

    train_dataset = Wav2LipSyncNetDataset(args.data_root, 'train', hparams, syncnet_T, syncnet_mel_step_size)
    test_dataset = FullVideoSyncDataset(args.data_root, 'val', hparams, syncnet_T, syncnet_mel_step_size, wrong_only = False)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers)
    val_subset_size = int(limit_val_batches * len(test_dataset))
    val_indices = torch.randperm(len(test_dataset))[:val_subset_size]
    val_subset = Subset(test_dataset, val_indices)
    test_data_loader = DataLoader(val_subset, batch_size=1, num_workers=num_workers, shuffle=False)

    # Model
    model = LipSyncExpertModel(hparams, syncnet_T, syncnet_mel_step_size, args.checkpoint_path)

    # Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('./checkpoints', args.checkpoint_dir),
        filename='checkpoint_{epoch:02d}_{step:02d}',
        save_top_k=10,
        monitor='train_loss',
        mode='min'
    )
    
    accelerator = args.accelerator
    if accelerator == 'gpu':
        #strategy = "ddp_find_unused_parameters_true"
        strategy = 'auto'
        devices = args.devices
        torch.compile(model.model)
    else:
        strategy = 'auto'
        devices = None
    
    # Trainer setup
    trainer = Trainer(
        max_epochs = hparams.nepochs,
        limit_train_batches=args.limit_train_batches,
        accelerator = accelerator,
        devices = devices,
        strategy = strategy,
        log_every_n_steps=4,
        logger = logger,
        precision = args.precision,              # Enable FP16 mixed precision training
        callbacks = [checkpoint_callback],
        check_val_every_n_epoch = eval_every # Validation after every N epoch
    )

    # Training
    trainer.fit(model, train_dataloaders=train_data_loader, val_dataloaders=test_data_loader)

if __name__ == "__main__":
    main()