import os

import torch
from torch.utils.data import DataLoader,Subset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from Wav2LipMarlin.hparams import hparams
from Wav2LipPLModel import Wav2LipPLModel
from SyncNetPLModel import LipSyncExpertModel
from Wav2LipMarlin.datasets.dataset import Wav2LipDataset
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_lightning import seed_everything



# Create a TensorBoard logger
logger = TensorBoardLogger(
    save_dir='./logs/lightning_logs',   # Directory where logs will be saved
    name='wav2lip_training',            # Name of the experiment (log folder name)
)

def main():
    # Argument Parsing
    import argparse
    parser = argparse.ArgumentParser(description='Train Lip-Sync Expert Model')
    parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)
    parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
    parser.add_argument('--syncnet_checkpoint_path', help='pre-trained LipSync expert checkpoint; required for training', required=True, type=str)
    parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', default=None, type=str)
    parser.add_argument('--accelerator', help='Device type', default="gpu", type=str)
    parser.add_argument('--devices', help='Device ids', default=1, type=int)
    parser.add_argument('--eval_every', help='Eval every N epochs', default=None, type=int)
    parser.add_argument('--num_workers', help='Device ids', default=None, type=int)
    parser.add_argument('--sync_weight', help='Sync loss weight', default=None, type=float)
    parser.add_argument('--is_weight', help='Identity-Style loss weight', default=None, type=float)
    parser.add_argument('--batch_size', help='Batch size for training', default=None, type=int)
    parser.add_argument('--limit_train_batches', help='Loop training for some batches', default=None, type=int)
    parser.add_argument('--precision', help='Loop training for some batches', default="32", type=str) # '16-mixed' for fp16
    parser.add_argument('--seed', help='Seed everything with provided value; -1 for a random seed', default=666, type=int)
    parser.add_argument('--wav2lip_type', 
        help='Combination of Wav2Lip and SyncExpert backends: original==fully original, marlin_full==both use MARLIN as video encoder; marlin==SyncNet original and Wav2Lip+MARLIN', 
        default="marlin", type=str)
    
    args = parser.parse_args()

    # Setup parameters
    num_workers = args.num_workers if args.num_workers is not None else hparams.num_workers
    batch_size = args.batch_size if args.batch_size is not None else hparams.batch_size
    eval_every = args.eval_every if args.eval_every is not None else 1
    eval_steps = 700 # from original Wav2Lip code
    
    if args.seed != -1:
        seed_everything(args.seed, workers = True) # Seed everything including workers
        
    if args.wav2lip_type == 'marlin_full':
        # Params for Syncnet-Marlin and Wav2Lip-Marlin
        syncnet_T = 5
        syncnet_T_m = int(3.2 * syncnet_T)
        
        syncnet_mel_step_size = 16
        syncnet_mel_step_size_m = int(syncnet_mel_step_size * 3.18)

        # both are MARLIN-based
        syncnet_mel_step_size = syncnet_mel_step_size_m
        syncnet_T = syncnet_T_m
        
        wav2lip_type='marlin'
        syncnet_type = 'marlin'
        hparams.img_size = 224
    elif args.wav2lip_type == 'original':
        # Params for Syncnet-orginal and Wav2Lip-original
        syncnet_T = 5
        syncnet_T_m = syncnet_T
        
        syncnet_mel_step_size = 16
        syncnet_mel_step_size_m = syncnet_mel_step_size
        
        wav2lip_type = 'original'
        syncnet_type = 'original'
        hparams.img_size = 96
    elif args.wav2lip_type == 'marlin':
        # Params for Syncnet-orginal and Wav2Lip-Marlin
        syncnet_T = 5
        syncnet_mel_step_size = 16
        
        syncnet_T_m = int(3.2 * syncnet_T)       
        syncnet_mel_step_size_m = int(syncnet_mel_step_size * 3.18)
        
        wav2lip_type = 'marlin'
        syncnet_type = 'original'
        hparams.img_size = 224    
    
    if args.sync_weight is not None:   
        hparams.set_hparam('syncnet_wt', args.sync_weight)
    
    if args.is_weight is not None:
        hparams.set_hparam('identity_style_wt', args.is_weight)
        from Wav2LipMarlin.loss.identity_style_loss import IdentityStyleLoss
        identity_style_loss = IdentityStyleLoss()
    else:
        hparams.set_hparam('identity_style_wt', 0.)
        identity_style_loss = None
    
    # syncnet_T_m = 5
    # syncnet_mel_step_size_m = 16
    train_dataset = Wav2LipDataset(args.data_root, 'train', hparams, syncnet_T_m, syncnet_mel_step_size)
    test_dataset = Wav2LipDataset(args.data_root, 'val', hparams, syncnet_T_m, syncnet_mel_step_size)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers)
    val_subset_size = eval_steps
    val_indices = torch.randperm(len(test_dataset))[:val_subset_size]
    val_subset = Subset(test_dataset, val_indices)
    test_data_loader = DataLoader(val_subset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # LipSync-expert model
    syncnet_pl_model = LipSyncExpertModel(hparams, syncnet_T, syncnet_mel_step_size, args.syncnet_checkpoint_path, 
            syncnet_type = syncnet_type)
    syncnet_pl_model.setup() # pre-load model
    # Model
    model = Wav2LipPLModel(hparams, syncnet_T, syncnet_mel_step_size, syncnet_pl_model, args.checkpoint_path, 
        wav2lip_type=wav2lip_type)

    # Checkpointing callbacks
    checkpoint_callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('./checkpoints/wav2lip_marlin/', args.checkpoint_dir),
        filename='l1_tr_checkpoint_{epoch:02d}_{step:02d}',
        save_top_k=10,
        monitor='L1_loss_train_epoch',
        mode='min'
    )
    checkpoint_callbacks.append(checkpoint_callback)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('./checkpoints/wav2lip_marlin/', args.checkpoint_dir),
        filename='sl_checkpoint_{epoch:02d}_{step:02d}',
        save_top_k=10,
        monitor='Sync_Loss_val_epoch',
        mode='min'
    )
    checkpoint_callbacks.append(checkpoint_callback)
    
    devices = args.devices
    accelerator = args.accelerator
    if accelerator == 'gpu':
        strategy = "ddp_find_unused_parameters_true" if devices > 1 else 'auto'
        devices = args.devices
        torch.compile(model.model)
        torch.compile(syncnet_pl_model.model)
    else:
        strategy = 'auto'
        devices = None
        
    if identity_style_loss is not None:
        model.setup()
        model.set_identity_style_loss(identity_style_loss)
    
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
        callbacks = checkpoint_callbacks,
        check_val_every_n_epoch = eval_every     # Validation after every N epoch
    )

    # Training
    trainer.fit(model, train_dataloaders=train_data_loader, val_dataloaders=test_data_loader)

if __name__ == "__main__":
    main()