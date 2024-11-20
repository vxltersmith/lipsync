import torch
from torch import nn
from torch.nn import functional as F
import math

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

class Wav2Lip(nn.Module):
    def __init__(self):
        super(Wav2Lip, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)), # 96,96

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 48,48
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 24,24
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 12,12
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 6,6
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # 3,3
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),
            
            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0), # 1, 1
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),])

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0),),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0), # 3,3
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),), # 6, 6

            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),), # 12, 12

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),), # 24, 24

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),), # 48, 48

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),),]) # 96,96

        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()) 

    def forward(self, audio_sequences, face_sequences):
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences) # B, 512, 1, 1

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e
            
            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2)  # (B, C, T, H, W)

        else:
            outputs = x
            
        return outputs
    
class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, heads=8):
        super(CrossAttention, self).__init__()
        self.attn = nn.MultiheadAttention(dim_q, heads)
        self.proj_kv = nn.Linear(dim_kv, dim_q)
        self.norm = nn.LayerNorm(dim_q)
    
    def forward(self, q, kv):
        # q: [B, C, H, W], kv: [B, C_audio, 1, 1]
        B, C_q, H, W = q.shape
        q = q.view(B, C_q, H * W).permute(2, 0, 1)  # [N, B, C_q]
        
        # Reshape kv for cross-attention (audio embedding is 1x1)
        kv = kv.view(B, kv.shape[1]).unsqueeze(0)  # [1, B, C_kv] if kv is global
        
        q = self.attn(q, kv, kv)[0]
        q = q.permute(1, 2, 0).view(B, C_q, H, W)  # Reshape back to original
        return q

from marlin_pytorch import Marlin
# Wav2Lip_Marlin
class Wav2Lip_Marlin(nn.Module):
    def __init__(self, backbone = "marlin_vit_base_ytf", 
            marlin_ckpt ='./checkpoints/MARLIN/marlin_vit_base_ytf.encoder.pt', 
            marlin_out_dim=768, external_pooler = True, dropout = 0.05, spectral_scaling = True, 
            mean_context = False):
        super(Wav2Lip_Marlin, self).__init__()
        # Modified MARLIN encoder; changed from original
        self.face_encoder = MARLIN_Encoder_decorator(backbone, marlin_ckpt, dropout = dropout, mean_context = mean_context)

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1, dropout=0),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True, dropout=0),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True, dropout=0),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1, dropout=0),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, dropout=0),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, dropout=dropout),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1, dropout=dropout),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, dropout=dropout),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, dropout=dropout),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1, dropout=dropout),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True, dropout=dropout),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0, dropout=dropout),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0, dropout=dropout),
        )

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0),),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0), #3,3
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True, dropout=0, spectral_scaling=spectral_scaling),),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True, dropout=0, spectral_scaling=spectral_scaling),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True, dropout=0, spectral_scaling=spectral_scaling),), # 6, 6

            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True, dropout=0, spectral_scaling=spectral_scaling),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True, dropout=0, spectral_scaling=spectral_scaling),), # 12, 12

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True, dropout=0, spectral_scaling=spectral_scaling),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True, dropout=0, spectral_scaling=spectral_scaling),), # 24, 24

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, dropout=0, spectral_scaling=spectral_scaling),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, dropout=0, spectral_scaling=spectral_scaling),), # 48, 48

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, dropout=0, spectral_scaling=spectral_scaling),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, dropout=0, spectral_scaling=spectral_scaling),),]) # 96,96

        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1, dropout=0, spectral_scaling=spectral_scaling),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()) 

    def forward(self, audio_sequences, face_sequences):
        """
         audio_sequences = (B, T, 1, 80, 16) \n
         face_sequences = (B, C, T, 224, 224) MARLIN input dims
        """
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        audio_input_dim_size = len(audio_sequences.size())
        assert input_dim_size == audio_input_dim_size
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            # face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        # Extract audio features with shape [B, 512, 1, 1]
        audio_embedding = self.audio_encoder(audio_sequences)
        # Extract list of a video features from MARLIN-encoder
        feats = self.face_encoder(face_sequences)
        # Compute image decoder 
        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e
            
            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            # x_split = torch.split(x, B, dim=0) # [(B, C, H, W)]
            # outputs = torch.stack(x_split, dim=2) # (B, C, T, H, W)
            x = x.reshape(B, -1, 3, 96, 96) # [B, T, C, H, W]
            outputs = x.transpose(1, 2) #[ B, C, T, H, W]
        else:
            outputs = x
            
        return outputs

class TemporalExpansionLayer(nn.Module):
    def __init__(self, input_dims, intermediate_dims, output_dims, dropout=0.1):
        super(TemporalExpansionLayer, self).__init__()
        # Initial block processes 8 frames, then we upsample to 16
        self.initial_conv = nn.Conv3d(input_dims, intermediate_dims, kernel_size=3, padding=1)  # Example 3D conv to process temporal data
        # Temporal Upsampling layers
        self.temporal_upsample = nn.ConvTranspose3d(intermediate_dims, 
            intermediate_dims, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)
        )
        self.dropout = nn.Dropout3d(p=dropout)
        self.conv1 = nn.Conv3d(intermediate_dims, output_dims, kernel_size=3, padding=1)
        self.activation = nn.GELU()

    def forward(self, x):
        # x is expected to have shape [batch_size, channels, frames, height, width]
        x = self.initial_conv(x)              # Process initial 8-frame input
        x = self.dropout(x)
        # Temporal Upsampling: Upsample from 8 to 16 frames
        x = self.temporal_upsample(x)        # Upsamples temporal dim from 8 to 16
        x = self.activation(self.conv1(x))    # Follow up with a convolution for refinement
        return x

class MarlinFeatureAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, interpolation_op=None, dropout = 0.05):
        super(MarlinFeatureAdapter, self).__init__()
               
        # Fusion convolution layers
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout)
        # Define normalization and activation
        self.layer_norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        
        # Interpolation operation (if defined)
        self.interpolation_op = interpolation_op
        
        # Define temporal expansion layer
        self.temporal_expansion_layer = TemporalExpansionLayer(
            input_dims=out_channels, 
            intermediate_dims=out_channels, 
            output_dims=out_channels,
            dropout = dropout * 2
        )

    def forward(self, x , batch_size:int):
        """
        x.shape = [B*T, H, W, C]
        """
        x = x.permute(0, 3, 1, 2)  # Change to [B*T, C, H, W]
        
        # Interpolation to change spatial scale
        if self.interpolation_op:
            x = self.interpolation_op(x)  # Rescale H, W according to level
        
        # conv-projection to fuse hiddens
        x = self.conv(x)  # [B*T, out_channels, H, W]
        x = self.dropout(x)
        x = x.permute(0, 2, 3, 1)  # [B*T, H, W, out_channels] for LayerNorm
        x = self.layer_norm(x)  # Normalize across channels
        x = x.permute(0, 3, 1, 2)  # Back to [B*T, out_channels, H, W]
        x = self.activation(x)  # Apply activation function
        
        # Temporal expansion x2 from T=8 to T=16
        C, H, W = x.shape[1:]
        x = x.reshape(batch_size, -1, C, H, W) # Back to [B, T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4) # Permute to [B, C, T, H, W]
        x = self.temporal_expansion_layer(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous() # Permute to [B, T, C, H, W]
        x = x.reshape(-1, C, H, W) # Back to [B*T, out_channels, H, W]
        
        return x

class InterpolationLayer(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=True):
        super(InterpolationLayer, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners if mode == 'bilinear' else None # Only used for modes like 'bilinear'

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor, 
                             mode=self.mode, align_corners=self.align_corners)
    
class MARLIN_Encoder_decorator(nn.Module):
    def __init__(self, 
            backbone, 
            marlin_ckpt, 
            seq_mean_pool = False,
            marlin_out_dims = 768, 
            dropout = 0.05,
            mean_context = False
        ):
        super(MARLIN_Encoder_decorator, self).__init__()
        self.encoder = Marlin.from_file(backbone, marlin_ckpt).encoder
        for p in self.encoder.parameters(): # Excluding syncnet weights from autograd
            p.requires_grad = False
        self.seq_mean_pool = seq_mean_pool
        self.mean_context = mean_context
        
        # Instantiate the list of align convs with interpolations attached, needed to match original architecture 
        self.channel_align_convs = nn.ModuleList([
            MarlinFeatureAdapter(marlin_out_dims*2, 512, nn.AdaptiveAvgPool2d((1, 1)), dropout = dropout),     # 512, 1, 1
            MarlinFeatureAdapter(marlin_out_dims*2, 512, nn.AdaptiveAvgPool2d((3, 3)), dropout = dropout),     # 512, 3, 3
            MarlinFeatureAdapter(marlin_out_dims*2, 256, nn.AdaptiveAvgPool2d((6, 6)), dropout = dropout),     # 256, 6, 6
            MarlinFeatureAdapter(marlin_out_dims*2, 128, nn.AdaptiveMaxPool2d((12, 12)), dropout = dropout),   # 128, 12, 12
            MarlinFeatureAdapter(marlin_out_dims*2, 64, InterpolationLayer(size=(24, 24), mode = 'bilinear'), dropout = dropout), # 64, 24, 24
            MarlinFeatureAdapter(marlin_out_dims*2, 32, InterpolationLayer(size=(48, 48), mode = 'bilinear'), dropout = dropout), # 32, 48, 48
            MarlinFeatureAdapter(marlin_out_dims*2, 16, InterpolationLayer(size=(96, 96), mode = 'bilinear'), dropout = dropout)  # 16, 96, 96
        ])
        
    def extract_features_ae(self, x):
        encoder = self.encoder
        batch_size, channels, T_, _,_ = x.shape
        x = encoder.patch_embedding(x)
        x = encoder.pos_embedding(x)
        
        feats = []
        for i, block in enumerate(encoder.blocks):
            x = block(x)
            if i % 2 == 0: continue
            # B, 8 *14 * 14, 768
            feats.append(x)

        if self.seq_mean_pool:
            x = x.mean(dim=1)
            x = encoder.norm(x)
            x = x.unsqueeze(-1).unsqueeze(-1)
        feats.append(x)
        return feats
    
    def forward(self, x):
        """
        Shape sequence of original encoder
            x = B, C, T, 224, 224
        Original shapes of Wav2Lip U-Net encoder  
            f0 = 16, 96,96
            f1 = 64, 48,48
            f2 = 64, 24,24
            f3 = 128, 12,12
            f4 = 256, 6,6
            f5 = 512, 3,3
            x_out = 512, 1, 1
        """
        # Extract original feature-maps with MARLIN-encoder
        x_masked = x[:,:3]
        x_reference = x[:,3:]
        with torch.no_grad():
            masked_feats = self.extract_features_ae(x_masked)
            reference_feats = self.extract_features_ae(x_reference)
        # Adaption to Wav2Lip default U-NET decoder
        feats = []
        for i in range(len(masked_feats)):
            masked_f_map = masked_feats[-1]
            reference_f_map = reference_feats[-1]
            
            # Average reference_f_map over temporal dimension
            reference_f_map_avg = reference_f_map.mean(dim=1, keepdim=True)  # Shape: [B, 1, H, W, C/2]
            reference_f_map_avg = self.encoder.norm(reference_f_map_avg)
            # Expand averaged reference features to match temporal dimension
            reference_f_map_avg_expanded = reference_f_map_avg.expand(-1, masked_f_map.shape[1], -1)  # Shape: [B, T, H, W, C/2]
            if not self.mean_context:
                # Concatenate masked_f_map with expanded averaged reference features
                reference_f_map = reference_f_map_avg_expanded + reference_f_map
                reference_f_map = self.encoder.norm(reference_f_map)
            else:
                reference_f_map = reference_f_map_avg_expanded
            
            feat = torch.cat([masked_f_map, reference_f_map], dim=-1)
            batch_size = feat.size(0)
            feat = feat.reshape(batch_size, 8, 14, 14, -1).contiguous()
            T = feat.shape[1]
            feat = feat.view(batch_size*T, feat.size(2), feat.size(3), feat.size(4)) # Reshape to B*T, H, W, C
            feat = self.channel_align_convs[i](feat, batch_size) 
            feats.append(feat)
            masked_feats.pop()
            reference_feats.pop()
        
        feats.reverse()
        return feats

class Wav2Lip_disc_qual(nn.Module):
    def __init__(self):
        super(Wav2Lip_disc_qual, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(nonorm_Conv2d(3, 32, kernel_size=7, stride=1, padding=3)), # 48,96

            nn.Sequential(nonorm_Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2), # 48,48
            nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2), # 24,24
            nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2), # 12,12
            nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # 6,6
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),

            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1), # 3,3
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),),
            
            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=0), # 1,1
            nonorm_Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),])

        self.binary_pred = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        self.label_noise = .0

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2)//2:]

    def to_2d(self, face_sequences):
        B = face_sequences.size(0)
        face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        return face_sequences

    def perceptual_forward(self, false_face_sequences):
        false_face_sequences = self.to_2d(false_face_sequences)
        false_face_sequences = self.get_lower_half(false_face_sequences)

        false_feats = false_face_sequences
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)

        false_pred_loss = F.binary_cross_entropy(self.binary_pred(false_feats).view(len(false_feats), -1), 
                                        torch.ones((len(false_feats), 1)).cuda())

        return false_pred_loss

    def forward(self, face_sequences):
        face_sequences = self.to_2d(face_sequences)
        face_sequences = self.get_lower_half(face_sequences)

        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)

        return self.binary_pred(x).view(len(x), -1)