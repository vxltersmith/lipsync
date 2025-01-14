import os
import yaml
import argparse

import torch
from torch.utils.data import DataLoader, Subset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything

from Wav2LipMarlin.hparams import hparams
from Wav2LipPLModel import Wav2LipPLModel
from SyncNetPLModel import LipSyncExpertModel
from Wav2LipMarlin.datasets.dataset import Wav2LipDataset, Wav2LipPCMDataset

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(description='Train Lip-Sync Expert Model')
    parser.add_argument("--config", help="Path to the YAML configuration file", required=True, type=str)
    parser.add_argument("--override", nargs='*', help="Override specific config parameters using key=value pairs", default=None)
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    experiment_conf = config['experiment']
    dataset_conf = config['dataset']
    training_conf = config['training']
    model_conf = config['model']
    

    # Override config with command-line arguments
    if args.override:
        for override in args.override:
            key, value = override.split("=", 1)
            config[key] = yaml.safe_load(value)  # Parse value as YAML for correct type
    
    experiment_name = experiment_conf['name']
    # Create a TensorBoard logger
    logger = TensorBoardLogger(
        save_dir = './logs/lightning_logs/wav2lip_marlin_training',
        name = experiment_name,
    )
    samples_path = os.path.join('./logs/samples/', experiment_name)
    eval_every = experiment_conf.get('eval_every', 1)
    if 'seed' in experiment_conf:
        seed_everything(experiment_conf['seed'], workers=True)

    # Temporal parameters, important for base version and dataset configuration
    wav2lip_type = model_conf['wav2lip_type']
    if wav2lip_type == 'marlin_full':
        syncnet_T = 16
        syncnet_T_m = int(3.2 * syncnet_T)

        syncnet_mel_step_size = 16
        syncnet_mel_step_size_m = int(syncnet_mel_step_size * 3.18)

        syncnet_T = syncnet_T_m
        syncnet_mel_step_size = syncnet_mel_step_size_m

        wav2lip_type = 'marlin'
        syncnet_type = 'marlin'
        hparams.img_size = 224
    elif wav2lip_type == 'original':
        syncnet_T = 5
        syncnet_T_m = syncnet_T

        syncnet_mel_step_size = 16
        syncnet_mel_step_size_m = syncnet_mel_step_size

        wav2lip_type = 'original'
        syncnet_type = 'original'
        hparams.img_size = 96
    elif wav2lip_type == 'marlin':
        syncnet_T = 5
        syncnet_mel_step_size = 16

        syncnet_T_m = int(3.2 * syncnet_T)
        syncnet_mel_step_size_m = int(syncnet_mel_step_size * 3.18)

        wav2lip_type = 'marlin'
        syncnet_type = 'original'
        hparams.img_size = 224

    # Set-up datasets
    val_data_root = config['data_root'] if config['val_data_root'] is None else config['val_data_root']   
    dataset_type = dataset_conf.get('dataset_type', 'wav2lip_mel')
    num_workers = dataset_conf.get('num_workers', hparams.num_workers)
    batch_size = dataset_conf.get('batch_size', hparams.batch_size)
    eval_steps = dataset_conf.get('eval_steps', 700)
    if dataset_type == 'wav2lip_mel':
        train_dataset = Wav2LipDataset(config['data_root'], 'train', hparams, syncnet_T_m, syncnet_mel_step_size, manifest_root=config['manifest_root'])
        test_dataset = Wav2LipDataset(val_data_root, 'val', hparams, syncnet_T_m, syncnet_mel_step_size, manifest_root=config['manifest_root'])
    elif dataset_type == 'wav2lip_pcm':
        train_dataset = Wav2LipPCMDataset(config['data_root'], 'train', hparams, syncnet_T_m, syncnet_mel_step_size, manifest_root=config['manifest_root'])
        test_dataset = Wav2LipPCMDataset(val_data_root, 'val', hparams, syncnet_T_m, syncnet_mel_step_size, manifest_root=config['manifest_root'])

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    if eval_steps is not None:
        val_subset_size = eval_steps
        val_indices = torch.randperm(len(test_dataset))[:val_subset_size]
        test_dataset = Subset(test_dataset, val_indices)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)


    hparams.set_hparam('initial_learning_rate', training_conf.get('lr', hparams.initial_learning_rate))
    hparams.set_hparam('syncnet_wt', training_conf.get('syncnet_wt', hparams.syncnet_wt))

    # LipSync-expert model
    syncnet_pl_model = LipSyncExpertModel(hparams, syncnet_T, syncnet_mel_step_size, 
        config['syncnet_checkpoint_path'], syncnet_type=syncnet_type)
    syncnet_pl_model.setup()

    audio_encoder_type = config.get('audio_encoder_type', 'conv2d')
    # Model
    dropout = config.get('dropout', 0.05)
    spectral_scale = config.get('spectral_scale', True)
    mean_context = config.get('mean_context', False)
    load_optimizer = config.get('load_optimizer', True)
    weight_decay = config.get('weight_decay', 0.0)

    model = Wav2LipPLModel(
        hparams, syncnet_T, syncnet_mel_step_size, syncnet_pl_model, config.get('checkpoint_path', None),
        wav2lip_type=wav2lip_type, dropout=dropout, spectral_scale=spectral_scale, mean_context=mean_context,
        load_optimizer=load_optimizer, samples_dir=samples_path, audio_encoder_type=audio_encoder_type,
        deep_fusion=config['deep_fusion'], weight_decay=weight_decay
    )

    # Checkpointing callbacks
    checkpoint_callbacks = []
    checkpoints_dir = os.path.join('./checkpoints/wav2lip_marlin/', experiment_name)
    checkpoint_callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoints_dir,
            filename='l1_vl_checkpoint_{epoch:02d}_{step:02d}',
            save_top_k=10,
            monitor='L1_val',
            mode='min'
        )
    )
    checkpoint_callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoints_dir,
            filename='sl_checkpoint_{epoch:02d}_{step:02d}',
            save_top_k=10,
            monitor='Sync_Loss_val_epoch',
            mode='min'
        )
    )
    checkpoint_callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoints_dir,
            filename='total_best_checkpoint_{epoch:02d}_{step:02d}',
            save_top_k=10,
            monitor='Total_loss_val',
            mode='min'
        )
    )

    devices = config.get('devices', 1)
    accelerator = config.get('accelerator', 'gpu')

    if accelerator == 'gpu':
        strategy = "ddp_find_unused_parameters_true" if devices > 1 else 'auto'
        torch.compile(model.model)
        torch.compile(syncnet_pl_model.model)
    else:
        strategy = 'auto'

    if 'is_weight' in config:
        hparams.set_hparam('identity_style_wt', config['is_weight'])
        from Wav2LipMarlin.loss.identity_style_loss import IdentityStyleLoss
        identity_style_loss = IdentityStyleLoss()
    else:
        hparams.set_hparam('identity_style_wt', 0.)
        identity_style_loss = None
    if identity_style_loss is not None:
        model.setup()
        model.set_identity_style_loss(identity_style_loss)

    # Trainer setup
    trainer = Trainer(
        max_epochs=hparams.nepochs,
        limit_train_batches=config.get('limit_train_batches', None),
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        log_every_n_steps=config.get("log_every_n_steps",4),
        logger=logger,
        precision=config.get('precision', '32'),
        callbacks=checkpoint_callbacks,
        check_val_every_n_epoch=eval_every
    )

    # Training
    trainer.fit(model, train_dataloaders=train_data_loader, val_dataloaders=test_data_loader)

if __name__ == "__main__":
    main()