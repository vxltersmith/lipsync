from Wav2LipMarlin.models.wav2lip import Wav2Lip as Wav2Lip_Original
from Wav2LipMarlin.models.wav2lip import Wav2Lip_Marlin
from Wav2LipMarlin.utils import save_sample_videos, save_sample_images
from pytorch_lightning import LightningModule
from torch import nn, optim
import torch
import torch.nn.functional as F



def _load(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)  # or 'cuda' for GPU
    return checkpoint

class Wav2LipModel(LightningModule):
    def __init__(self, hparams, syncnet_T, syncnet_mel_step_size, syncnet_pl_model, checkpoint_path=None, samples_dir='./samples',
            wav2lip_type='marlin', loss_type='bce', scheduler = False):
        super(Wav2LipModel, self).__init__()
        self.config = hparams
        self.syncnet_T = syncnet_T
        self.syncnet_mel_step_size = syncnet_mel_step_size
        self.checkpoint_path = checkpoint_path
        self.scheduler = scheduler
        
        assert syncnet_pl_model is not None # it's important to have it
        self.syncnet_pl_model = syncnet_pl_model
        for p in syncnet_pl_model.model.parameters(): # Excluding syncnet weights from autograd
            p.requires_grad = False
        
        self.checkpoint = None
        self.samples_dir = samples_dir
               
        if wav2lip_type == 'marlin':
            self.model = Wav2Lip_Marlin()  # Redesigned Wav2Lip model with MARLIN encoder
        elif wav2lip_type == 'original':
            self.model = Wav2Lip_Original()
        else:
            raise ValueError(f"Unknown model type: {wav2lip_type}")
        self.wav2lip_type = wav2lip_type
        
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
            if self.wav2lip_type == "marlin":
                self.load_state_dict(self.checkpoint["state_dict"])
            elif self.wav2lip_type == "original":
                self.model.load_state_dict(self.checkpoint["state_dict"])
            else:
                raise NotImplementedError()
                
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
        
        syncnet_model = self.syncnet_pl_model.model
        # B, 3 * T, H//2, W || 3 * T == channels * temporal size
        a, v = syncnet_model(mel, g)
        
        y = torch.ones(g.size(0), 1).float().to(self.device)
        return self.cosine_loss(a, v, y)

    def sliding_window_loss(self, mel, g):
        """
        mel: full mel-spectrogram of the original clip
        g: a generated frames and croped lower half of faces
        
        output: averaged sync_loss over the temporal window longer that syncnet_T
        """
        syncnet_model = self.syncnet_pl_model.model
        
        batch_size = g.size(0)
        temporal_size = self.syncnet_T  # Temporal window size (T)
        frame_height = g.size(3)  # Height of the frame
        frame_width = g.size(4)   # Width of the frame
        
        g = g[:, :, :, g.size(3)//2:]
        # Sliding window approach (to make it fit model's expected input)
        losses = []
        # Adjust T2 (current temporal size) to T (desired temporal size)
        for start_idx in range(0, g.size(2) - temporal_size + 1, temporal_size):  # Sliding across time dimension
            # Select a slice of T frames (of size temporal_size)
            window_g = g[:, :, start_idx:start_idx + temporal_size, :, :]  # Shape B, C, T, H, W
            # Temporal concatenation: Convert from (B, C, T, H, W) to (B, C*T, H, W)
            window_g = torch.cat([window_g[:, :, i] for i in range(temporal_size)], dim=1)  # B, C*T, H, W
            # For each temporal window, compute the sync loss
            a, v = syncnet_model(mel, window_g)
            # We want the ground truth label to be 1 for positive pairs
            y = torch.ones(batch_size, 1).float().to(self.device)
            # Calculate the cosine loss for this window
            loss = self.cosine_loss(a, v, y)
            losses.append(loss)
        # Average the losses over all windows
        avg_loss = torch.mean(torch.stack(losses))
        return avg_loss

    def get_mse_loss(self, g, gt):
        # Resize the generated image (g) to match the ground truth (gt) size if necessary
        if g.shape[3:] < gt.shape[3:]:
            batch_size, C, T, H, W = g.shape
            g = g.permute(0, 2, 1, 3, 4).contiguous()  # [batch_size, T, C, H, W]
            g = F.interpolate(g.view(-1, C, H, W), size=gt.shape[3:], mode='bilinear', align_corners=False)
            g = g.view(batch_size, T, C, *gt.shape[3:]).permute(0, 2, 1, 3, 4)  # [batch_size, C, T, new_H, new_W]
        
        # Compute L1 loss
        l1loss = self.recon_loss(g, gt)
        return l1loss, g

    def training_step(self, batch, batch_idx):
        # Training loop logic
        x, indiv_mels, mel, gt = batch
        # Preprocess input
        # MARLIN dims (B, C, T, H, W)
        # Original video input is a lower half of a face, here it's concatenated with blank other half 
        # to match MARLIN input H,W
        g = self.model(indiv_mels, x)

        syncnet_wt = self.config.syncnet_wt
        if syncnet_wt > 0.:
            sync_loss = self.get_sync_loss(mel, g)
        else:
            sync_loss = 0. 

        l1loss, g = self.get_mse_loss(g, gt)

        loss = syncnet_wt * sync_loss + (1 - syncnet_wt) * l1loss

        if self.global_step % self.config.checkpoint_interval == 0:
            save_sample_videos(x, g, gt, self.global_step, self.samples_dir)
            save_sample_images(x, g, gt, self.global_step, self.samples_dir)

        # Logging
        self.log('L1_loss_train', l1loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if syncnet_wt > 0.:
            self.log('Sync_Loss_train', sync_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation loop logic
        x, indiv_mels, mel, gt = batch

        g = self.model(indiv_mels, x)

        sync_loss = self.sliding_window_loss(mel, g)
        l1loss, g = self.get_mse_loss(g, gt)
       
        # Logging
        self.log('L1_val', l1loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('Sync_Loss_val', sync_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return sync_loss

    def on_validation_epoch_end(self):
        # Access the averaged epoch loss directly, as PL logs them automatically
        averaged_sync_loss = self.trainer.callback_metrics.get("Sync_Loss_val")
        averaged_recon_loss = self.trainer.callback_metrics.get("L1_val")

        # Optionally print or take further action based on the averaged losses
        print(f"Epoch End - L1: {averaged_recon_loss}, Sync Loss: {averaged_sync_loss}")

        # Adjust the sync weight if needed
        if averaged_sync_loss and averaged_sync_loss < 0.75:
            self.config.set_hparam('syncnet_wt', 0.01)
    
    def predict_step(self, batch, batch_idx):
        # Validation loop logic
        x, mel, y = batch
        return x

    def configure_optimizers(self):
        # Optimizer configuration
        optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad],
                               lr=self.config.initial_learning_rate)
        
        def load_checkpoint(checkpoint, optimizer, reset_optimizer = False):
            if not reset_optimizer:
                optimizer_state = checkpoint["optimizer_states"][0]
                if optimizer_state is not None:
                    print("Load optimizer state from state dict")
                    optimizer.load_state_dict(optimizer_state)
            return optimizer
        
        if self.checkpoint is not None:
            optimizer = load_checkpoint(self.checkpoint, optimizer)
        
        if self.scheduler:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(optimizer, 1e8, 1e-10)

            # Step 3: Return a dictionary with optimizer and scheduler
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_loss'  # Optional: monitor metric to decide when to update scheduler in some cases
            }
        
        return optimizer