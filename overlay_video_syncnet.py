import cv2
import torch
import torch.nn as nn
import ffmpeg
import numpy as np

import scipy.io.wavfile as wavfile
import tempfile

from Wav2LipMarlin.hparams import hparams
import Wav2LipMarlin.datasets.audio as audio
from SyncNetPLModel import LipSyncExpertModel

import Wav2LipMarlin.face_detection as face_detection

# Utility function to read video frames
def load_video_frames(video_path, img_size=96):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (img_size, img_size))  # Resize to model input size
        frame = frame[:, :, ::-1]  # Convert BGR (OpenCV) to RGB
        frames.append(frame)
    cap.release()
    video_tensor = torch.tensor(frames).permute(3, 0, 1, 2).float() / 255.0  # Shape: C, T, H, W
    return video_tensor.unsqueeze(0)  # Shape: 1, C, T, H, W

# Utility function to compute the mel spectrogram for audio
def load_mel_spectrogram(audio_path, config):
    wav = audio.load_wav(audio_path, config.sample_rate)
    mel = audio.melspectrogram(wav, config).T
    mel = torch.FloatTensor(mel.T).unsqueeze(0)
    return mel, wav

def crop_audio_window(spec, start_frame_num, syncnet_mel_step_size, config):
    # Calculate frames per second (mel frames per video frame)
    mel_frames_per_video_frame = config.sample_rate / (config.fps * config.hop_size)
    
    # Calculate the precise start and end indices in the mel spectrogram
    start_idx = int(start_frame_num * mel_frames_per_video_frame)
    end_idx = start_idx + syncnet_mel_step_size

    # Check if end index exceeds the length of the mel spectrogram
    if end_idx > spec.size(0):
        # Handle by replicating last frame
        sliced_spec = spec[start_idx:]
        repeat_count = syncnet_mel_step_size - sliced_spec.size(0)
        repeated_frames = sliced_spec[-1:].repeat(repeat_count, 1)
        padded_spec = torch.cat([sliced_spec, repeated_frames], dim=0)
    else:
        padded_spec = spec[start_idx:end_idx, :]

    return padded_spec


def evaluate_video_sync(video_frames, mel, syncnet_model, device):
    """
    Evaluates synchronization scores across an entire video using non-overlapping windows.
    """
    sync_scores = []
    temporal_size = syncnet_model.syncnet_T  # Expected window size
    syncnet_mel_step_size = syncnet_model.syncnet_mel_step_size
    config = syncnet_model.config

    for start_idx in range(0, video_frames.size(2) - temporal_size + 1, temporal_size):
        window_g = video_frames[:, :, start_idx:start_idx + temporal_size, :, :]
        window_g = torch.cat([window_g[:, :, i] for i in range(temporal_size)], dim=1)
        window_g = window_g[:,:,window_g.size(2)//2:,:]
        window_g = window_g.to(device)
        
        mel_c = crop_audio_window(mel[0].T, start_idx, syncnet_mel_step_size, config).T
        mel_c = mel_c.unsqueeze(0).unsqueeze(0)
        mel_c = mel_c.to(device)
        
        if mel_c.shape[-1] < syncnet_mel_step_size:
            sync_scores.append(torch.zeros(1))
            continue

        a, v = syncnet_model(mel_c, window_g)
        sync_score = nn.functional.cosine_similarity(a, v)
        sync_scores.append(sync_score)

    return sync_scores

def overlay_sync_scores(video_path, sync_scores, output_path, temporal_size, fps=25):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    score_idx = 0  # Track which sync score we are on
    frame_count = 0  # Frame counter to manage which window/frame we are on

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Display sync score for each frame in the window
        if score_idx < len(sync_scores):
            # Display the sync score for the current window
            score_text = f"Sync Score: {sync_scores[score_idx].item():.3f}"
            cv2.putText(frame, score_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Update frame_count to move to the next window after `temporal_size` frames
            if frame_count % temporal_size == 0:
                score_idx += 1

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Annotated video saved to {output_path}")


def combine_video_audio_ffmpeg(video_path, audio_path, output_path):
    input_video = ffmpeg.input(video_path)
    input_audio = ffmpeg.input(audio_path)
    ffmpeg.output(input_video, input_audio, output_path, vcodec='libx264', acodec='aac').run()
    
def remove_temp_files(video_path, audio_path):
    import os
    if os.path.exists(video_path):
        os.remove(video_path)
        print(f"Temporary video file {video_path} deleted.")
    if os.path.exists(audio_path):
        os.remove(audio_path)
        print(f"Temporary audio file {audio_path} deleted.")

# Function to save the audio array as a temporary wav file
def save_wav(wav_array, sample_rate=16000, output_path=None):
    """
    Saves the numpy array `wav_array` as a .wav file.
    The array is expected to be float32, so we scale and convert it to int16.
    """
    # Scale the audio array to the range of int16
    wav_array = np.int16(wav_array * 32767)
    
    # Default to saving as a temporary file if no output path is provided
    if output_path is None:
        # Create a temporary file to save the wav file
        temp_wav = tempfile.mktemp(suffix=".wav")
        output_path = temp_wav
    # Save to the provided output path
    wavfile.write(output_path, sample_rate, wav_array)
    
    print(f"Audio saved to {output_path}")
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate SyncNet on dataset')
    parser.add_argument("--video_path", required=True, help="Path to input video file")
    parser.add_argument('--checkpoint_path', help='Checkpoint path for SyncNet model', required=True)
    args = parser.parse_args()
    
    face_detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, 
                                    device='cuda')
    batch_size = 16
    
    syncnet_T=5
    syncnet_mel_step_size=16
    #hparams.img_size = 96
    
    probe = ffmpeg.probe(args.video_path)["streams"][0] 
    fps = float(probe['r_frame_rate'].split('/')[0]) / float(probe['r_frame_rate'].split('/')[1])
    hparams.set_hparam('fps', fps)
    
    n_frames = int(probe["nb_frames"])

    # Load video frames
    video_frames = load_video_frames(args.video_path, hparams.img_size)
    # Crop faces from video 
    video_frames_copy = video_frames.clone() * 255
    batches = [video_frames_copy.squeeze(0).permute(1,2,3,0)[i:i + batch_size] for i in range(0, n_frames, batch_size)]
    i = -1
    cropped_video_frames = []
    for fb in batches:
        preds = face_detector.get_detections_for_batch(np.asarray(fb))
        for j, f in enumerate(preds):
            i += 1
            if f is None:
                continue
            x1, y1, x2, y2 = f
            frame = fb[j][y1:y2, x1:x2]
            cropped_video_frames.append(frame)
            continue
        continue
    
    assert len(cropped_video_frames) == video_frames.shape[2]
    
    hparams.img_size = 96
    cropped_video_frames = torch.stack(
        [torch.nn.functional.interpolate(_.permute(2,0,1).unsqueeze(0), size=(hparams.img_size, hparams.img_size)) for _ in cropped_video_frames]
    )
    cropped_video_frames = cropped_video_frames.transpose(1, 0)
    cropped_video_frames = cropped_video_frames.transpose(2, 1)
    cropped_video_frames = cropped_video_frames.float() / 255.0 
    
    mel, wav = load_mel_spectrogram(args.video_path, hparams)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    syncnet_model = LipSyncExpertModel(hparams, syncnet_T=syncnet_T, syncnet_mel_step_size=syncnet_mel_step_size, 
                                       checkpoint_path=args.checkpoint_path, syncnet_type='original').to(device)
    syncnet_model.setup()
    syncnet_model.eval()

    sync_scores = evaluate_video_sync(cropped_video_frames, mel, syncnet_model, device)
    temp_video_path = "output_video_with_scores.mp4"
    overlay_sync_scores(args.video_path, sync_scores, temp_video_path, temporal_size=syncnet_T)
    temp_audio_path = save_wav(wav, hparams.sample_rate)
    combine_video_audio_ffmpeg(temp_video_path,  temp_audio_path, "final_video_with_audio.mp4")
    remove_temp_files(temp_video_path, temp_audio_path)

if __name__ == "__main__":
    main()