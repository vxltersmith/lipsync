import os
import cv2



def read_video_as_numpy(folder_path, video_name):
    # Construct the full video path
    video_path = os.path.join(folder_path, video_name)
    # Open the video using cv2
    cap = cv2.VideoCapture(video_path)
    # Check if video opened successfully
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open the video: {video_path}")
    frames = []  # List to hold the frames as numpy arrays
    # Read frames one by one
    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video or failed to read a frame
            break
        # Append frame to the list (as numpy array)
        frames.append(frame)
    # Release the video capture object
    cap.release()
    return frames

import cv2
import torch
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip

def save_video_with_audio(image_tensor, audio_tensor, fps=25, audio_sample_rate=44100, video_path='output_video.mp4', audio_path='output_audio.wav'):
    """
    Save a video with corresponding audio from two tensors: one for images and one for audio.
    
    Args:
        image_tensor (torch.Tensor): A tensor of shape (B, H, W, C) representing a batch of frames (RGB images).
        audio_tensor (torch.Tensor): A 1D tensor representing audio waveform.
        fps (int): Frames per second for the video.
        audio_sample_rate (int): Sample rate of the audio.
        video_path (str): The path where the video will be saved.
        audio_path (str): The path where the audio will be saved temporarily.
    """
    # Step 1: Convert the image tensor to numpy and write video frames using OpenCV
    B, H, W, C = image_tensor.shape  # B: number of frames, H: height, W: width, C: channels
    
    # Define video writer using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))
    
    # Convert the image tensor to numpy and write frames
    for i in range(B):
        frame = image_tensor[i].cpu().numpy().astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        video_writer.write(frame)
    
    video_writer.release()  # Release the video writer

    # Step 2: Save the audio tensor to a .wav file using scipy
    from scipy.io.wavfile import write
    audio_data = audio_tensor.cpu().numpy()  # Convert audio tensor to numpy
    write(audio_path, audio_sample_rate, audio_data)

    # Step 3: Use moviepy to combine video and audio
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    final_clip = video_clip.set_audio(audio_clip)
    
    # Step 4: Save the final video with audio
    final_clip.write_videofile("final_video_with_audio.mp4", codec='libx264', audio_codec='aac')

# # Example usage with random data
# image_tensor = torch.randint(0, 255, (100, 480, 640, 3), dtype=torch.uint8)  # 100 frames, 480x640 resolution, 3 channels (RGB)
# audio_tensor = torch.randn(44100 * 10)  # 10 seconds of audio (at 44.1 kHz)

# save_video_with_audio(image_tensor, audio_tensor, fps=25)
