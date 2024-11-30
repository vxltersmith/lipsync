from Wav2LipMarlin.models.wav2lip import Wav2Lip as Wav2Lip_Original
from Wav2LipMarlin.models.wav2lip import Wav2Lip_Marlin
from Wav2LipMarlin.utils import save_sample_gifs, save_sample_images
from pytorch_lightning import LightningModule
from torch import nn, optim
import torch
import torch.nn.functional as F



def _load(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)  # or 'cuda' for GPU
    return checkpoint

class Wav2LipPLModel(LightningModule):
    def __init__(self, hparams, syncnet_T, syncnet_mel_step_size, syncnet_pl_model, checkpoint_path=None, samples_dir='./samples',
            wav2lip_type='marlin', scheduler = False, sync_offset_frames = 2, infer_only = False,
            dropout = 0.05, spectral_scale = True, mean_context = False, load_optimizer = False, 
            resize_outputs = False, audio_encoder_type = 'default'):
        super(Wav2LipPLModel, self).__init__()
        self.config = hparams
        self.syncnet_T = syncnet_T
        self.syncnet_mel_step_size = syncnet_mel_step_size
        self.checkpoint_path = checkpoint_path
        self.scheduler = scheduler
        
        self.infer_only = infer_only
        if not infer_only:
            assert syncnet_pl_model is not None # it's important to have it
            self.syncnet_pl_model = syncnet_pl_model
            for p in syncnet_pl_model.model.parameters(): # Excluding syncnet weights from autograd
                p.requires_grad = False
        
        self.checkpoint = None
        self.load_optimizer = load_optimizer
        self.samples_dir = samples_dir
               
        if wav2lip_type == 'marlin':
            self.model = Wav2Lip_Marlin(dropout=dropout, spectral_scaling=spectral_scale, # Redesigned Wav2Lip model with MARLIN encoder
                mean_context = mean_context, audio_encoder_type = audio_encoder_type)  
        elif wav2lip_type == 'original':
            self.model = Wav2Lip_Original()
        else:
            raise ValueError(f"Unknown model type: {wav2lip_type}")
        
        self.wav2lip_type = wav2lip_type
        self.sync_offset_frames = sync_offset_frames
        
        self.identity_style_loss_fn = None
        self.loaded = False
        
        self.resize_outputs = resize_outputs
        self.audio_encoder_type = audio_encoder_type
        
    def setup(self, stage=None):
        if self.loaded: return
        
        def load_checkpoint(path):
            print("Load checkpoint from: {}".format(path))
            checkpoint = _load(path, self.device)
            return checkpoint
        if self.checkpoint_path is not None:
            self.checkpoint = load_checkpoint(self.checkpoint_path)
            
            if self.infer_only:
                # delete syncnet keys from checkpoint if any
                for _ in [_ for _ in self.checkpoint['state_dict'] if 'syncnet_pl_model' in _]:
                    del self.checkpoint['state_dict'][_]
            # identity_style_loss_fn
            for _ in [_ for _ in self.checkpoint['state_dict'] if 'identity_style_loss_fn' in _]:
                del self.checkpoint['state_dict'][_]
            
            if self.wav2lip_type == "marlin":
                self.load_state_dict(self.checkpoint["state_dict"])
            elif self.wav2lip_type == "original":
                self.model.load_state_dict(self.checkpoint["state_dict"])
            else:
                raise NotImplementedError()
            self.loaded = True
                
        self.logloss = nn.BCELoss()
        self.recon_loss = nn.L1Loss()
        
    def set_identity_style_loss(self, identity_style_loss_fn):
        self.identity_style_loss_fn = identity_style_loss_fn

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
        mel: concatenated slices of original mel with length mel-T and shape [B, syncnet_T, 1, 80, mel_T]
        g: a generated frames and croped lower half of faces
        
        output: averaged sync_loss over the temporal window longer that syncnet_T
        """
        syncnet_model = self.syncnet_pl_model.model
        
        batch_size = g.size(0)
        temporal_size = self.syncnet_T  # Temporal window size (T)
        frame_T= g.size(2)
        frame_height = g.size(3)  # Height of the frame
        frame_width = g.size(4)   # Width of the frame
        
        syncnet_frame_height = frame_height // 2 # needed to get a lower half of face crop
        
        g = g[:, :, :, syncnet_frame_height:]
        # Sliding window approach (to make it fit model's expected input)
        losses = []
        # Adjust T2 (current temporal size) to T (desired temporal size)
        for start_idx in range(0, frame_T - temporal_size, temporal_size):  # Sliding across time dimension
            # Select a slice of T frames (of size temporal_size)
            video_window_g = g[:, :, start_idx:start_idx + temporal_size, :, :]  # Shape B, C, T, H, W
            # Temporal concatenation: Convert from (B, C, T, H, W) to (B, C*T, H, W)
            video_window_g = video_window_g.reshape(batch_size, 3 * temporal_size, syncnet_frame_height, frame_width)  # B, C*T, H, W
            
            #Select correspinding mel-spec slice of mel-T length
            # In dataloader mels aligned with current_frame - 2, so for a current window we get full slice shifted to the right
            mel_window = mel[:, start_idx+self.sync_offset_frames , :, :]
            
            # For each temporal window, compute the sync loss
            a, v = syncnet_model(mel_window, video_window_g)
            # Calculate the cosine loss for this window
            loss = (1 - nn.functional.cosine_similarity(a, v)).mean()
            losses.append(loss)
        
        # To process last chunk
        if frame_T % temporal_size != 0: 
            start_idx = frame_T - temporal_size
            # Select a slice of T frames (of size temporal_size)
            video_window_g = g[:, :, start_idx:start_idx + temporal_size, :, :]  # Shape B, C, T, H, W
            # Temporal concatenation: Convert from (B, C, T, H, W) to (B, C*T, H, W)
            video_window_g = video_window_g.reshape(batch_size, 3 * temporal_size, syncnet_frame_height, frame_width)  # B, C*T, H, W
            
            #Select correspinding mel-spec slice of mel-T length
            # In dataloader mels aligned with current_frame - 2, so for a current window we get full slice shifted to the right
            mel_window = mel[:, start_idx+self.sync_offset_frames, :, :]
            
            # For each temporal window, compute the sync loss
            a, v = syncnet_model(mel_window, video_window_g)
            # Calculate the cosine loss for this window
            loss = (1 - nn.functional.cosine_similarity(a, v)).mean()
            losses.append(loss)
        # Average the losses over all windows
        avg_loss = torch.mean(torch.stack(losses))
        return avg_loss

    def get_mse_loss(self, g, gt):
        # Resize the generated image (g) to match the ground truth (gt) size if necessary
        if g.shape[3:] < gt.shape[3:]:
            batch_size, C, T, H, W = g.shape #[ B, C, T, H, W]
            g = g.permute(0, 2, 1, 3, 4).contiguous()  # [batch_size, T, C, H, W]
            g = F.interpolate(g.view(-1, C, H, W), size=gt.shape[3:], mode='bilinear', align_corners=False)
            g = g.view(batch_size, T, C, *gt.shape[3:]).permute(0, 2, 1, 3, 4)  # [batch_size, C, T, new_H, new_W]
        
        # Compute L1 loss
        l1loss = self.recon_loss(g, gt)
        return l1loss, g

    def training_step(self, batch, batch_idx):
        # Training loop logic
        original = self.audio_encoder_type == 'default'
        syncnet_wt = self.config.syncnet_wt
        
        if original: # should work for mels
            x, indiv_mels, mel, gt = batch
            # Preprocess input
            # MARLIN dims (B, C, T, H, W)
            # Original video input is a lower half of a face, here it's concatenated with blank other half 
            # to match MARLIN input H,W
            g = self.model(indiv_mels, x)
            
            if syncnet_wt > 0.:
                sync_loss = self.sliding_window_loss(indiv_mels, g)
            else:
                sync_loss = 0.
        else: # for PCM-encoded audio case
            x, audio_window, gt = batch
            g = self.model(audio_window, x)
            #TODO figure out syncloss for PCM window
            sync_loss = 0.

        l1loss, g = self.get_mse_loss(g, gt)
        if self.global_rank == 0 and self.global_step % self.config.checkpoint_interval == 0:
            save_sample_gifs(x, g, gt, self.global_step, self.samples_dir)
        
        if self.identity_style_loss_fn is not None:
            batch_size, C, T, H, W = g.shape
            g = g.transpose(2,1).contiguous().reshape(batch_size*T, C, H, W)
            gt = gt.transpose(2,1).contiguous().reshape(batch_size*T, C, H, W)
            identity_syle_loss = self.identity_style_loss_fn(g, gt)
            is_weight = self.config.identity_style_wt
        else:
            is_weight = 0

        loss =  (1 - syncnet_wt - is_weight) * l1loss + syncnet_wt * sync_loss + is_weight * identity_syle_loss

        # Logging
        self.log('L1_loss_train', l1loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if syncnet_wt > 0.:
            self.log('Sync_Loss_train', sync_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if is_weight > 0:
            self.log('Identity_Style_Loss_train', identity_syle_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        self.log('Total_loss_train', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        original = self.audio_encoder_type == 'default'
        if original: # should work for mels
            # Validation loop logic
            x, indiv_mels, mel, gt = batch
            g = self.model(indiv_mels, x)
            sync_loss = self.sliding_window_loss(indiv_mels, g)
        else: # for PCM-encoded audio case
            x, audio_window, gt = batch
            g = self.model(audio_window, x)
            #TODO figure out syncloss for PCM window
            sync_loss = torch.zeros(1).to(x.device)
        
        l1loss, g = self.get_mse_loss(g, gt)
        
        if batch_idx == 0 and self.global_rank == 0:
            import os
            samples_val_path = os.path.join(self.samples_dir, 'eval')
            save_sample_gifs(x, g, gt, self.global_step, samples_val_path)
        
        if self.identity_style_loss_fn is not None:
            batch_size, C, T, H, W = g.shape
            g = g.transpose(2,1).contiguous().reshape(batch_size*T, C, H, W)
            gt = gt.transpose(2,1).contiguous().reshape(batch_size*T, C, H, W)
            identity_syle_loss = self.identity_style_loss_fn(g, gt)
            is_weight = self.config.identity_style_wt
        else:
            is_weight = 0
        
        syncnet_wt = self.config.syncnet_wt
        loss = syncnet_wt * sync_loss + (1 - syncnet_wt) * l1loss
        if is_weight > 0:
            loss = is_weight * identity_syle_loss + (1-is_weight) * loss
       
        # Logging
        self.log('L1_val', l1loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('Sync_Loss_val', sync_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if is_weight > 0:
            self.log('Identity_Style_Loss_val', identity_syle_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('Total_loss_val', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
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
        x, indiv_mels = batch
        with torch.no_grad():
            g = self.model(indiv_mels, x)
        
        if self.resize_outputs:
            batch_size, C, T, H, W = g.shape #[ B, C, T, H, W]
            g = g.permute(0, 2, 1, 3, 4).contiguous()  # [batch_size, T, C, H, W]
            g = F.interpolate(g.view(-1, C, H, W), size=x.shape[3:], mode='bilinear', align_corners=False)
            g = g.view(batch_size, T, C, *x.shape[3:]).permute(0, 2, 1, 3, 4)  # [batch_size, C, T, new_H, new_W]
        
        return g

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
        
        if self.checkpoint is not None and self.load_optimizer:
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