from Wav2LipMarlin.models.syncnet import SyncNet_color as SyncNet
from pytorch_lightning import LightningModule
from torch import nn, optim
import torch


def _load(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)  # or 'cuda' for GPU
    return checkpoint

class LipSyncExpertModel(LightningModule):
    def __init__(self, hparams, syncnet_T, syncnet_mel_step_size, checkpoint_path=None):
        super(LipSyncExpertModel, self).__init__()
        self.config = hparams
        self.syncnet_T = syncnet_T
        self.syncnet_mel_step_size = syncnet_mel_step_size
        self.checkpoint_path = checkpoint_path
        self.checkpoint = None
        
        self.model = SyncNet()  # Original SyncNet model
        self.loss_fn = nn.BCELoss()
        #self.loss_fn = nn.CosineEmbeddingLoss(0.0)
        
    def setup(self, stage=None):
        def load_checkpoint(path):
            print("Load checkpoint from: {}".format(path))
            checkpoint = _load(path, self.device)
            return checkpoint
        if self.checkpoint_path is not None:
            self.checkpoint = load_checkpoint(self.checkpoint_path)
            self.load_state_dict(self.checkpoint["state_dict"])

    def forward(self, mel, x_full):
        # Forward pass through the SyncNet model
        a, v = self.model(mel, x_full)
        return a, v

    def cosine_loss(self, a, v, y):
        # Cosine similarity loss
        # y = y*2 - 1 # to labels rescale to -1:1
        #loss = self.loss_fn(a,v,y.squeeze(-1))
        
        # Original Wav2Lip-expert loss
        d = nn.functional.cosine_similarity(a, v)
        d = d+1/2 
        loss = self.loss_fn(d.unsqueeze(1), y)
        return loss

    def training_step(self, batch, batch_idx):
        # Training loop logic
        x, mel, y = batch

        # Preprocess input
        # marlin dims (B, C, T, H, W)
        x = x.reshape(x.size(0), -1, 3, x.shape[-2], x.shape[-1]).transpose(2, 1)
        x_full = torch.cat([torch.zeros_like(x), x], dim=-2)

        a, v = self(mel, x_full)  # Forward pass

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
        loss = self.cosine_loss(a, v, y) + 0.5 * nn.functional.mse_loss(a, v)
        # Logging
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        # Validation loop logic
        x, mel, y = batch
        x = x.squeeze(0) # batch size should be 1
        mel = mel.squeeze(0) # batch size should be 1
        # Preprocess input (B, C, T, H, W)
        x_full = torch.cat([torch.zeros_like(x), x], dim=-2)
        audio_embedding, video_embedding = self(mel, x_full)  # Forward pass
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

