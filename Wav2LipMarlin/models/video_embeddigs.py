import torch
import torch.nn as nn
import torch.nn.functional as F
from marlin_pytorch import Marlin



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
            model_name, 
            checkpoint_path = None, 
            seq_mean_pool = False,
            marlin_out_dims = 768, 
            dropout = 0.05,
            mean_context = False
        ):
        super(MARLIN_Encoder_decorator, self).__init__()
        marlin_model = Marlin.from_file(model_name, checkpoint_path) if checkpoint_path is not None else Marlin.from_online(model_name)
        self.encoder = marlin_model.encoder
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
    
    def get_emb_for_loss(self, x):
        encoder = self.encoder
        x = encoder.patch_embedding(x)
        x = encoder.pos_embedding(x)
        return encoder.blocks[0](x)
        
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