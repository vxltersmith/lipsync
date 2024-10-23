import torch
from torch.utils.data import DataLoader,Subset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from Wav2LipMarlin.hparams import hparams
from SyncNetPLModel import LipSyncExpertModel
from Wav2LipMarlin.datasets.dataset import Wav2LipDataset
from pytorch_lightning.loggers import TensorBoardLogger




# Create a TensorBoard logger
logger = TensorBoardLogger(
    save_dir='./logs/lightning_logs',   # Directory where logs will be saved
    name='training',                    # Name of the experiment (log folder name)
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
    parser.add_argument('--num_workers', help='Device ids', default=None, type=int)
    parser.add_argument('--limit_train_batches', help='Loop training for some batches', default=None, type=int)
    parser.add_argument('--precision', help='Loop training for some batches', default="32", type=str) # '16-mixed' for fp16

    args = parser.parse_args()

    # Dataset and Dataloader setup
    syncnet_T = 16
    syncnet_mel_step_size = int(16 * 3.18)
    num_workers = args.num_workers if args.num_workers is not None else hparams.num_workers

    train_dataset = Wav2LipDataset(args.data_root, 'train', hparams, syncnet_T, syncnet_mel_step_size)
    test_dataset = Wav2LipDataset(args.data_root, 'val', hparams, syncnet_T, syncnet_mel_step_size)

    train_data_loader = DataLoader(train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True,
                                   num_workers=num_workers)
    val_subset_size = int(0.25 * len(test_dataset))
    val_indices = torch.randperm(len(test_dataset))[:val_subset_size]
    val_subset = Subset(test_dataset, val_indices)
    test_data_loader = DataLoader(val_subset, batch_size=hparams.syncnet_batch_size, num_workers=num_workers)

    # Model
    model = LipSyncExpertModel(hparams, syncnet_T, syncnet_mel_step_size, args.checkpoint_path)

    # Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='checkpoint_{epoch:02d}_{step:02d}',
        save_top_k=3,
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
        logger = logger,
        precision = args.precision,              # Enable FP16 mixed precision training
        callbacks = [checkpoint_callback],
        check_val_every_n_epoch = 2 if args.limit_train_batches is None else int(1e3)  # Validation after every N epoch
    )

    # Training
    trainer.fit(model, train_dataloaders=train_data_loader, val_dataloaders=test_data_loader)

if __name__ == "__main__":
    main()