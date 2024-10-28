from Wav2LipMarlin.models.syncnet import SyncNet_Marlin
from Wav2LipMarlin.models.syncnet import SyncNet_color as SyncNet_Original
from pytorch_lightning import LightningModule
from torch import nn, optim
import torch


def _load(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)  # or 'cuda' for GPU
    return checkpoint

class LipSyncExpertModel(LightningModule):
    def __init__(self, hparams, syncnet_T, syncnet_mel_step_size, checkpoint_path=None, 
            syncnet_type='marlin', loss_type='bce'):
        super(LipSyncExpertModel, self).__init__()
        self.config = hparams
        self.syncnet_T = syncnet_T
        self.syncnet_mel_step_size = syncnet_mel_step_size
        self.checkpoint_path = checkpoint_path
        self.checkpoint = None
        
        if syncnet_type == 'marlin':
            self.model = SyncNet_Marlin()  # Redesigned SyncNet model
        elif syncnet_type == 'original':
            self.model = SyncNet_Original()
        else:
            raise ValueError(f"Unknown model type: {syncnet_type}")
        self.syncnet_type = syncnet_type
        
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

    def forward(self, mel, x_full):
        # Forward pass through the SyncNet model
        a, v = self.model(mel, x_full)
        return a, v

    def cosine_loss(self, a, v, y):
        a = a.cpu()
        v = v.cpu()
        y = y.cpu()
        # Cosine similarity loss
        if self.loss_type == 'cosine_embedding':
            y = y*2 - 1 # to labels rescale to -1:1
            loss = self.loss_fn(a,v,y.squeeze(-1))
        # Original Wav2Lip-expert loss
        if self.loss_type == 'bce':
            d = nn.functional.cosine_similarity(a, v)
            d = (d+1)/2 
            loss = self.loss_fn(d.unsqueeze(1), y)
        return loss

    def training_step(self, batch, batch_idx):
        # Training loop logic
        x, mel, y = batch
        # Preprocess input
        # MARLIN dims (B, C, T, H, W)
        # Original video input is a lower half of a face, here it's concatenated with blank other half 
        # to match MARLIN input H,W
        x = x.reshape(x.size(0), -1, 3, x.shape[-2], x.shape[-1]).transpose(2, 1)
        x_full = torch.cat([torch.zeros_like(x), x], dim=-2)
        # Forward pass
        a, v = self(mel, x_full)
        # Compute loss
        loss = self.cosine_loss(a, v, y)
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation loop logic
        x, mel, y = batch

        # Preprocess input
        x = x.reshape(x.size(0), -1, 3, x.shape[-2], x.shape[-1]).transpose(2, 1)
        x_full = torch.cat([torch.zeros_like(x), x], dim=-2)

        a, v = self(mel, x_full)  # Forward pass

        # Compute loss
        loss = self.cosine_loss(a, v, y) # + 0.5 * nn.functional.mse_loss(a, v)
        # Logging
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        # Validation loop logic
        x, mel, y = batch
        x = x.squeeze(0) # batch size should be 1
        mel = mel.squeeze(0) # batch size should be 1

        if self.syncnet_type == 'marlin':
            # Preprocess input (B, C, T, H, W)
            x_full = torch.cat([torch.zeros_like(x), x], dim=-2)
        else:
            # SyncNet-original input is (B, C*T, H, W)
            x_full = x.transpose(2,1)
            x_full = x_full.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        # Forward pass
        audio_embedding, video_embedding = self(mel, x_full)
        # Cosine similarity
        sync_scores = torch.nn.functional.cosine_similarity(video_embedding, audio_embedding, dim=-1)
        y = y * 2 -1
        cos_loss = torch.nn.functional.cosine_embedding_loss(video_embedding, audio_embedding, y[0])
        sync_scores = sync_scores.cpu().mean()
        print(f'Step {batch_idx}, score: {sync_scores}, loss: {cos_loss}')
        return sync_scores

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
        return optimizer

