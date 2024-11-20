import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

class InceptionResnetV1WithIntermediates(InceptionResnetV1):
    """Wrapper to extract intermediate activations from InceptionResnetV1."""
    def forward(self, x):
        """
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        """
        # Forward pass but capturing intermediate outputs
        x_in = self.conv2d_1a(x)
        x1 = self.conv2d_2a(x_in)
        x2 = self.conv2d_2b(x1)
        x3 = self.maxpool_3a(x2)
        
        x4 = self.conv2d_3b(x3)
        x5 = self.conv2d_4a(x4)
        x6 = self.conv2d_4b(x5)
        
        # Normal forward pass to get final embedding
        final_embedding = super().forward(x)
        return final_embedding, [x_in, x1, x2, x3, x4, x5, x6]

class IdentityStyleLoss(torch.nn.Module):
    def __init__(self, resnet=None, alpha = 0.3, betta = 0.7, style_layers = [1, 2, 3], identity_layers = [3, 4]):
        super(IdentityStyleLoss, self).__init__()
        
        # Load a pre-trained model for feature extraction (e.g., FaceNet)
        self.resnet = resnet if resnet else InceptionResnetV1WithIntermediates(pretrained='vggface2').eval()
        for param in self.resnet.parameters():
            param.requires_grad = False  # Freeze the model
        
        self.alpha = alpha
        self.betta = betta
        self.style_layers = style_layers  # Earlier layers for style
        self.identity_layers = identity_layers  # Later layers for identity

    def gram_matrix(self, feature_map):
        """Compute the Gram matrix for style loss."""
        (B, C, H, W) = feature_map.size()
        features = feature_map.view(B, C, H * W)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (C * H * W)

    def forward(self, generated, target):
        # Pass generated and target images through the model to get feature maps
        with torch.no_grad():
            identity_gen, gen_features = self.resnet(generated)
            identity_target, target_features = self.resnet(target)

        style_loss = 0
        identity_loss = 0

        # Accumulate style loss from earlier layers
        for i, (gen_feat, tgt_feat) in enumerate(zip(gen_features, target_features)):
            if i in self.style_layers:
                gram_gen = self.gram_matrix(gen_feat)
                gram_tgt = self.gram_matrix(tgt_feat)
                style_loss += F.mse_loss(gram_gen, gram_tgt)

            # Accumulate identity loss from later layers
            if i in self.identity_layers:
                identity_loss += F.mse_loss(gen_feat, tgt_feat)
        
        identity_loss += F.mse_loss(identity_gen, identity_target)

        # Weight the style and identity components and return the total loss
        total_loss = self.alpha * style_loss + self.betta * identity_loss
        return total_loss
