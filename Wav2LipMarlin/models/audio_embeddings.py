import torch
import torch.nn as nn
import Wav2LipMarlin.datasets.audio as audio

from transformers import Wav2Vec2Model



class Wav2Vec2FeatureAdapter(nn.Module):
    def __init__(self, model_name = 'facebook/wav2vec2-large-xlsr-53', hidden_size=1024, 
            select_from_layers = [3, 5, 6, 9, 12, 15, 18, 21], temporal_window_frames=16):
        """
            This module extracts features from PCM-encoded audio sequence and prepares it for U-Net fusion
            Shape sequence of original Wav2Lip2.0
                x = B, T, hidden_size=1024
            Original shapes of Wav2Lip U-Net encoder  
                f0 = 16, 96,96
                f1 = 64, 48,48
                f2 = 64, 24,24
                f3 = 128, 12,12
                f4 = 256, 6,6
                f5 = 512, 3,3
                x_out = 512, 1, 1
        """
        super(Wav2Vec2FeatureAdapter, self).__init__()
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.conv_projectors = nn.ModuleList([
            nn.Conv1d(hidden_size, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(hidden_size, 512 * 2 , kernel_size=3, stride=1, padding=1),
            nn.Conv1d(hidden_size, 512 * 2, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(hidden_size, 512 * 2, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(hidden_size, 384 * 2, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(hidden_size, 256 * 2, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(hidden_size, 128 * 2, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(hidden_size, 64 * 2, kernel_size=3, stride=1, padding=1),
        ])
        self.select_from_layers = select_from_layers
        self.temporal_window_frames = temporal_window_frames
        return
    
    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, audio_input):
        batch_size = audio_input.size(0)
        with torch.no_grad():
            audio_features = self.model(audio_input, output_hidden_states=True)
        audio_features = list(audio_features.hidden_states)
        
        filtered_audio_features = [audio_features[i] for i in self.select_from_layers]
        
        adapted_audio_features = []       
        for i in range(len(filtered_audio_features)):
            feat = filtered_audio_features[-1]
            feat = feat.transpose(2, 1)
            feat = self.conv_projectors[i](feat)
            feat = nn.functional.adaptive_avg_pool1d(feat, 16)
            feat = feat.reshape(batch_size, -1, self.temporal_window_frames, 1, 1)
            feat = feat.transpose(2, 1).contiguous()
            feat = feat.reshape(self.temporal_window_frames * batch_size, -1, 1, 1) # Back to [B*T, out_channels, H, W]
            adapted_audio_features.append(feat)
            filtered_audio_features.pop()
        
        adapted_audio_features.reverse()
        return adapted_audio_features
        
if __name__ == '__main__':
    audio_path = './target_audio.wav'

    sr = 16000
    fps = 25
    v_win = 16

    context_length = v_win / fps
    a_win = int(context_length*sr)

    audio_input = audio.load_wav(audio_path, sr)
    audio_input = torch.tensor(audio_input).unsqueeze(0)
    audio_input = audio_input[:,:a_win]

    wav2vec = Wav2Vec2FeatureAdapter()

    with torch.no_grad():
        audio_features = wav2vec(audio_input)

    low_level = audio_features[0]  # Low-level
    mid_level = audio_features[3]  # Mid-level
    high_level = audio_features[-1]  # High-level

    print('done!')