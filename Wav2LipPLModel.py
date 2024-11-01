from Wav2LipMarlin.models.wav2lip import Wav2Lip as Wav2Lip_Original
from Wav2LipMarlin.models.wav2lip import Wav2Lip_Marlin
from Wav2LipMarlin.utils import save_sample_images
from pytorch_lightning import LightningModule
from torch import nn, optim
import torch
import numpy as np




def _load(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)  # or 'cuda' for GPU
    return checkpoint

class Wav2LipModel(LightningModule):
    def __init__(self, hparams, syncnet_T, syncnet_mel_step_size, syncnet_model, checkpoint_path=None, checkpoint_dir='./samples',
            wav2lip_type='marlin', loss_type='bce'):
        super(Wav2LipModel, self).__init__()
        self.config = hparams
        self.syncnet_T = syncnet_T
        self.syncnet_mel_step_size = syncnet_mel_step_size
        self.checkpoint_path = checkpoint_path
        self.syncnet_model = syncnet_model
        self.checkpoint = None
        self.checkpoint_dir = checkpoint_dir
               
        if wav2lip_type == 'marlin':
            self.syncnet = Wav2Lip_Marlin()  # Redesigned SyncNet model
        elif wav2lip_type == 'original':
            self.syncnet = Wav2Lip_Original()
        else:
            raise ValueError(f"Unknown model type: {wav2lip_type}")
        self.wav2lip_type = wav2lip_type
        
        for p in self.syncnet.parameters():
            p.requires_grad = False
        
        if loss_type == 'bce':
            self.loss_fn = nn.BCELoss()
        elif loss_type == 'cosine_embedding':
            self.loss_fn = nn.CosineEmbeddingLoss(0.0)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        self.loss_type = loss_type
        
    def setup(self, stage=None):
        def load_checkpoint(path):
            print("Load checkpoint from: {}".format(path))
            checkpoint = _load(path, self.device)
            return checkpoint
        if self.checkpoint_path is not None:
            self.checkpoint = load_checkpoint(self.checkpoint_path)
            if self.syncnet_type == "marlin":
                self.load_state_dict(self.checkpoint["state_dict"])
            elif self.syncnet_type == "original":
                self.model.load_state_dict(self.checkpoint["state_dict"])
                
        self.logloss = nn.BCELoss()
        self.recon_loss = nn.L1Loss()

    def forward(self, mel, x_full):
        # Forward pass through the SyncNet model
        a, v = self.model(mel, x_full)
        return a, v

    def cosine_loss(self, a, v, y):
        d = nn.functional.cosine_similarity(a, v)
        loss = self.logloss(d.unsqueeze(1), y)
        return loss

    def get_sync_loss(self, mel, g):
        g = g[:, :, :, g.size(3)//2:]
        g = torch.cat([g[:, :, i] for i in range(self.syncnet_T)], dim=1)
        # B, 3 * T, H//2, W
        a, v = self.syncnet_model(mel, g)
        y = torch.ones(g.size(0), 1).float().to(self.device)
        return self.cosine_loss(a, v, y)

    def training_step(self, batch, batch_idx):
        # Training loop logic
        x, mel, indiv_mels, gt = batch
        # Preprocess input
        # MARLIN dims (B, C, T, H, W)
        # Original video input is a lower half of a face, here it's concatenated with blank other half 
        # to match MARLIN input H,W
        g = self.model(indiv_mels, x)

        if self.hparams.syncnet_wt > 0.:
            sync_loss = self.get_sync_loss(mel, g)
        else:
            sync_loss = 0. 

        l1loss = self.recon_loss(g, gt)

        syncnet_wt = self.hparams.syncnet_wt
        loss = syncnet_wt * sync_loss + (1 - syncnet_wt) * l1loss

        if self.global_step % self.hparams.checkpoint_interval == 0:
            save_sample_images(x, g, gt, self.global_step, self.checkpoint_dir)

        if self.hparams.syncnet_wt > 0.:
            running_sync_loss += sync_loss.item()
        else:
            running_sync_loss += 0.

        # Logging
        self.log('L1: ', l1loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('Sync Loss: ', running_sync_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation loop logic
        x, indiv_mels, mel, gt = batch

        g = self.model(indiv_mels, x)

        sync_loss = self.get_sync_loss(mel, g)
        l1loss = self.recon_loss(g, gt)
       
        # Logging
        self.log('L1: ', l1loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('Sync Loss: ', sync_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return sync_loss

    def on_validation_epoch_end(self):
        # Access the averaged epoch loss directly, as PL logs them automatically
        averaged_sync_loss = self.trainer.callback_metrics.get("sync_loss")
        averaged_recon_loss = self.trainer.callback_metrics.get("L1_loss")

        # Optionally print or take further action based on the averaged losses
        print(f"Epoch End - L1: {averaged_recon_loss}, Sync Loss: {averaged_sync_loss}")

        # Adjust the sync weight if needed
        if averaged_sync_loss and averaged_sync_loss < 0.75:
            self.hparams.syncnet_wt = 0.01
    
    def predict_step(self, batch, batch_idx):
        # Validation loop logic
        x, mel, y = batch
        return x

    def configure_optimizers(self):
        # Optimizer configuration
        optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad],
                               lr=self.config.syncnet_lr)
        
        def load_checkpoint(checkpoint, optimizer, reset_optimizer = False):
            if not reset_optimizer:
                optimizer_state = checkpoint["optimizer_states"][0]
                if optimizer_state is not None:
                    print("Load optimizer state from state dict")
                    optimizer.load_state_dict(optimizer_state)
            return optimizer
        if self.checkpoint is not None:
            optimizer = load_checkpoint(self.checkpoint, optimizer)
        
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, 1e8, 1e-10)

        # Step 3: Return a dictionary with optimizer and scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'  # Optional: monitor metric to decide when to update scheduler in some cases
        }