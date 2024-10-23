#from MARLIN.evaluate import evaluate

from marlin_pytorch import Marlin
import torch

def main():
#    evaluate()

    backbone = "marlin_vit_base_ytf"
    marlin_ckpt ='./MARLIN/ckpt/marlin_vit_base_ytf.encoder.pt'
    model = Marlin.from_file(backbone, marlin_ckpt).encoder
    
    batch_size = 4
    t = 16
    channels = 3
    W=H=224
    
    model = model.cuda()
    # (B, C, T, H, W)
    sample_tensor = torch.rand(batch_size, channels, t, W, H)
    sample_tensor = sample_tensor.cuda()
    with torch.no_grad():
        features = model.extract_features(sample_tensor, False)
        features_w2l = features.reshape(batch_size*t, 14, 14, -1)
    return

if __name__ == '__main__':
    main()
    print('done!')
