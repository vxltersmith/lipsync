import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv2d
from marlin_pytorch import Marlin

class SyncNet_Marlin(nn.Module):
    def __init__(self, backbone = "marlin_vit_base_ytf", 
            marlin_ckpt ='./checkpoints/MARLIN/marlin_vit_base_ytf.encoder.pt', 
            marlin_out_dim=768, external_pooler = True):
        super(SyncNet_Marlin, self).__init__()
        # changed from original
        self.face_encoder = Marlin.from_file(backbone, marlin_ckpt).encoder
        for p in self.face_encoder.parameters():
            p.requires_grad = False
        
        self.external_pooler = external_pooler
        if external_pooler:
            self.pooler = ExternalAttentionPooler(marlin_out_dim, 512)

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
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            # changed from original
            Conv2d(512, 512, kernel_size=(1, 3), stride=1, padding=0),
            Conv2d(512, 512, kernel_size=(1, 3), stride=1, padding=0),
            Conv2d(512, 512, kernel_size=(1, 3), stride=1, padding=0),
            nn.Conv2d(512, 768, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, audio_sequences, face_sequences): # audio_sequences := (B, dim, T)
        with torch.no_grad():
            face_embedding = self.face_encoder.extract_features(face_sequences, not self.external_pooler)
        if self.external_pooler:
            face_embedding =  self.pooler(face_embedding)
        
        face_embedding = F.normalize(face_embedding, p=2, dim=1)
            
        audio_embedding = self.audio_encoder(audio_sequences)
        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)

        return audio_embedding, face_embedding

class SyncNet_color(nn.Module):
    def __init__(self):
        super(SyncNet_color, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

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
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, audio_sequences, face_sequences): # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)


        return audio_embedding, face_embedding
    
    
class ExternalAttentionPooler(nn.Module):
    def __init__(self, input_dim, attention_dim=128):
        super(ExternalAttentionPooler, self).__init__()
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, vit_features):
        # vit_features shape: (B, T, D)
        
        # Calculate attention weights
        attention_weights = self.attention(vit_features)  # Shape: (B, T, 1)
        attention_weights = F.softmax(attention_weights, dim=1)  # Normalize along T dimension

        # Weighted sum pooling
        pooled_features = (vit_features * attention_weights).sum(dim=1)  # Shape: (B, D)
        
        return pooled_features