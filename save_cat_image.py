from PIL import Image
import torchvision

tensor = x[0].permute(1, 0, 2, 3).cpu()

# Create a grid of images (for 16 frames, 4x4 grid makes sense)
grid_img = torchvision.utils.make_grid(tensor, nrow=4, padding=2)  # nrow=4 means 4 images per row

# Convert the grid tensor to a PIL image (this is needed for saving)
ndarr = grid_img.mul(255).byte().numpy()  # Convert tensor to numpy array
ndarr = ndarr.transpose(1, 2, 0)  # Transpose from CxHxW to HxWxC (needed for PIL)
im = Image.fromarray(ndarr)

# Save the image
im.save('stacked_frames_grid.png')