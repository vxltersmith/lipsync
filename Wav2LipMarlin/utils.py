from os.path import join, exists
from os import makedirs
import numpy as np
import cv2


def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    makedirs(folder, exist_ok=True)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])
            
            
def save_sample_videos(x, g, gt, global_step, checkpoint_dir, fps=25, video_filename_prefix='sample_video'):
    """
    Save each batch element as a separate video instead of individual image files.
    
    Args:
        x (torch.Tensor): Tensor for reference and input images of shape (B, T, C, H, W).
        g (torch.Tensor): Tensor for generated images of shape (B, T, C, H, W).
        gt (torch.Tensor): Tensor for ground truth images of shape (B, T, C, H, W).
        global_step (int): Global step for naming and organizing files.
        checkpoint_dir (str): Directory where videos will be saved.
        fps (int): Frames per second for the video.
        video_filename_prefix (str): Prefix for the video file name.
    """
    # Convert tensors to numpy arrays and rearrange dimensions
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    # Concatenate references, inputs, generated, and ground truth along width
    refs, inps = x[..., 3:], x[..., :3]
    collage = np.concatenate((refs, inps, g, gt), axis=-2)  # Shape (B, T, H, W, C)

    # Ensure the checkpoint directory exists
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    makedirs(folder, exist_ok=True)

    # Loop through each batch element and save as a separate video
    B, T, H, W, C = collage.shape  # B: batch size, T: time steps, H, W: height, width, C: channels
    for batch_idx in range(B):
        # Define video writer for each batch element
        video_path = join(folder, f"{video_filename_prefix}_batch{batch_idx}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))

        # Write each frame from the collage to the video file
        for t in range(T):
            frame = collage[batch_idx, t]
            frame_rgb = frame
            #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for saving
            video_writer.write(frame_rgb)
        
        video_writer.release()  # Release the video writer
        print(f"Video saved for batch {batch_idx} at {video_path}")
 
from PIL import Image
import numpy as np
import os

def save_sample_gifs(x, g, gt, global_step, checkpoint_dir, fps=25, gif_filename_prefix='sample_gif'):
    """
    Save each batch element as a separate GIF instead of individual image files.
    
    Args:
        x (torch.Tensor): Tensor for reference and input images of shape (B, T, C, H, W).
        g (torch.Tensor): Tensor for generated images of shape (B, T, C, H, W).
        gt (torch.Tensor): Tensor for ground truth images of shape (B, T, C, H, W).
        global_step (int): Global step for naming and organizing files.
        checkpoint_dir (str): Directory where GIFs will be saved.
        fps (int): Frames per second for the GIF.
        gif_filename_prefix (str): Prefix for the GIF file name.
    """
    # Convert tensors to numpy arrays and rearrange dimensions
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    # Concatenate references, inputs, generated, and ground truth along width
    refs, inps = x[..., 3:], x[..., :3]
    collage = np.concatenate((refs, inps, g, gt), axis=-2)  # Shape (B, T, H, W, C)

    # Ensure the checkpoint directory exists
    folder = os.path.join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    os.makedirs(folder, exist_ok=True)

    # Loop through each batch element and save as a separate GIF
    B, T, H, W, C = collage.shape  # B: batch size, T: time steps, H, W: height, width, C: channels
    duration = int(1000 / fps)  # Duration per frame in milliseconds for the GIF

    for batch_idx in range(B):
        # Define GIF path for each batch element
        gif_path = os.path.join(folder, f"{gif_filename_prefix}_batch{batch_idx}.gif")

        # Collect frames for the GIF
        frames = [Image.fromarray(cv2.cvtColor(collage[batch_idx, t], cv2.COLOR_BGR2RGB)) for t in range(T)]

        # Save frames as GIF
        frames[0].save(
            gif_path, 
            save_all=True, 
            append_images=frames[1:], 
            duration=duration, 
            loop=0
        )
        
        print(f"GIF saved for batch {batch_idx} at {gif_path}")       

def save_batch_gifs(g, global_step, checkpoint_dir, fps=25, gif_filename_prefix='sample_gif'):
    """
    Save each batch element as a separate GIF instead of individual image files.
    
    Args:
        x (torch.Tensor): Tensor for reference and input images of shape (B, T, C, H, W).
        g (torch.Tensor): Tensor for generated images of shape (B, T, C, H, W).
        gt (torch.Tensor): Tensor for ground truth images of shape (B, T, C, H, W).
        global_step (int): Global step for naming and organizing files.
        checkpoint_dir (str): Directory where GIFs will be saved.
        fps (int): Frames per second for the GIF.
        gif_filename_prefix (str): Prefix for the GIF file name.
    """
    # Convert tensors to numpy arrays and rearrange dimensions
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    # Concatenate references, inputs, generated, and ground truth along width
    refs, inps = g[..., 3:], g[..., :3]
    collage = g[0]  # Shape (B, T, H, W, C)

    # Ensure the checkpoint directory exists
    folder = checkpoint_dir
    os.makedirs(folder, exist_ok=True)

    # Loop through each batch element and save as a separate GIF
    #B, T, H, W, C = collage.shape  # B: batch size, T: time steps, H, W: height, width, C: channels
    duration = int(1000 / fps)  # Duration per frame in milliseconds for the GIF

    # Define GIF path for each batch element
    gif_path = os.path.join(folder, f"{gif_filename_prefix}_batch{global_step}.gif")
    T = collage.shape[0]
    # Collect frames for the GIF
    frames = [Image.fromarray(cv2.cvtColor(collage[t], cv2.COLOR_BGR2RGB)) for t in range(T)]

    # Save frames as GIF
    frames[0].save(
        gif_path, 
        save_all=True, 
        append_images=frames[1:], 
        duration=duration, 
        loop=0
    )
    
    print(f"GIF saved at {gif_path}")     
        
import os
import numpy as np
from glob import glob

def load_and_concatenate_all_batches(checkpoint_dir, global_step, output_filename):
    # Define the folder where images were saved
    folder = os.path.join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    
    # Get list of all saved images sorted by filename
    image_files = sorted(glob(os.path.join(folder, '*.jpg')))
    
    # Dictionary to group images by batch index
    images_by_batch = {}
    
    # Organize file paths by batch index
    for img_path in image_files:
        # Extract batch index and time step from filename (format is 'batchidx_timestep.jpg')
        filename = os.path.basename(img_path)
        batch_idx, timestep = map(int, filename.split('.')[0].split('_'))
        
        # Append to batch dictionary
        if batch_idx not in images_by_batch:
            images_by_batch[batch_idx] = []
        images_by_batch[batch_idx].append(img_path)
    
    # List to hold concatenated images for each batch
    concatenated_batches = []
    
    # Load, sort by timestep, and concatenate images for each batch
    for batch_idx, image_paths in sorted(images_by_batch.items()):
        # Sort image paths by timestep within each batch
        image_paths = sorted(image_paths, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        
        # Load images and concatenate them along the y-axis
        images = [cv2.imread(img_path) for img_path in image_paths]
        concatenated_image = np.concatenate(images, axis=0)
        
        # Add the concatenated batch to our list
        concatenated_batches.append(concatenated_image)
    
    # Concatenate all batches horizontally along the x-axis
    final_image = np.concatenate(concatenated_batches, axis=1)
    
    # Save the final concatenated image
    cv2.imwrite(output_filename, final_image)
    print("Final concatenated image saved to:", output_filename)
