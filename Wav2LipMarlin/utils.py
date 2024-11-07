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
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for saving
            video_writer.write(frame_rgb)
        
        video_writer.release()  # Release the video writer
        print(f"Video saved for batch {batch_idx} at {video_path}")
