import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.utils.spectral_norm as spectral_norm


def apply_spectral_norm(model):
    for name, layer in model.named_children():
        # Check if the layer is a Conv2d or Linear layer (Spectral Norm is applied to weights)
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            setattr(model, name, spectral_norm(layer))
        elif isinstance(layer, torch.nn.Sequential):
            # Recursively apply to submodules
            apply_spectral_norm(layer)

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, dropout = 0.05, 
            spectral_scaling = False, spectral_scaling_inject_later=False,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout),
            nn.Dropout2d(p=dropout)
        )
        self.act = nn.ReLU()
        self.residual = residual

        self.spectral_scaling = spectral_scaling

        if not spectral_scaling_inject_later and self.spectral_scaling:
            apply_spectral_norm(self.conv_block)
            self.spectral_scaling = False

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if self.spectral_scaling:
            apply_spectral_norm(self.conv_block)
            self.spectral_scaling = False
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)
