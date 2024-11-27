import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

from .video_embeddigs import MARLIN_Encoder_decorator
from .audio_embeddings import Wav2Vec2FeatureAdapter

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

# Wav2Lip_Marlin
class Wav2Lip_Marlin(nn.Module):
    def __init__(self, backbone = "marlin_vit_base_ytf", 
            marlin_ckpt ='./checkpoints/MARLIN/marlin_vit_base_ytf.encoder.pt', 
            marlin_out_dim=768, external_pooler = True, dropout = 0.05, spectral_scaling = True, 
            mean_context = False, audio_encoder_type = 'default'):
        super(Wav2Lip_Marlin, self).__init__()
        # Modified MARLIN encoder; changed from original
        self.face_encoder = MARLIN_Encoder_decorator(backbone, marlin_ckpt, dropout = dropout, mean_context = mean_context)

        if audio_encoder_type == 'default':
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
        else:
            self.audio_encoder = Wav2Vec2FeatureAdapter()
        self.audio_encoder_type = audio_encoder_type

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
        original = self.audio_encoder_type == 'default'
        if original:
            input_dim_size = len(face_sequences.size())
            audio_input_dim_size = len(audio_sequences.size())
            assert input_dim_size == audio_input_dim_size
            if input_dim_size > 4:
                audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            # Extract audio features with shape [B, 512, 1, 1]
        audio_embedding = self.audio_encoder(audio_sequences)
        
        # Extract list of a video features from MARLIN-encoder
        feats = self.face_encoder(face_sequences)
        # Compute image decoder
        if original:
            x = audio_embedding
        else:
            x = audio_embedding[-1]
            audio_embedding.pop()
        for f in self.face_decoder_blocks:
            x = f(x)
            
            if not original: # FiLM fusion of audio embeddings
                gamma, betta = audio_embedding[-1].chunk(2, dim=1)
                x = x * gamma + betta
                audio_embedding.pop()
            
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e
            
            feats.pop()

        x = self.output_block(x)

        if not original or input_dim_size > 4:
            # x_split = torch.split(x, B, dim=0) # [(B, C, H, W)]
            # outputs = torch.stack(x_split, dim=2) # (B, C, T, H, W)
            x = x.reshape(B, -1, 3, 96, 96) # [B, T, C, H, W]
            outputs = x.transpose(1, 2) #[ B, C, T, H, W]
        else:
            outputs = x
            
        return outputs

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