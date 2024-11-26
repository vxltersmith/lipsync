import ffmpeg
import torch

from Wav2LipPLModel import Wav2LipPLModel
from Wav2LipMarlin.hparams import hparams
from overlay_video_syncnet import save_wav, combine_video_audio_ffmpeg, remove_temp_files
import Wav2LipMarlin.face_detection as face_detection
from infer_wav2lip import get_smoothened_boxes
from Wav2LipMarlin.utils import save_batch_gifs
import numpy as np
import cv2

def get_mels(audio_path: str, mel_step_size: int, fps: int, config):
    print('Read audio and slice mels...')
    import Wav2LipMarlin.datasets.audio as audio
    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav, config)
    print(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80./fps 
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            last_chunk = mel[:, start_idx:-1]
            full_last_chunk = np.zeros((mel.shape[0], mel_step_size))
            full_last_chunk[:,:last_chunk.shape[-1]] = last_chunk
            mel_chunks.append(full_last_chunk)
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1
    print("Length of mel chunks: {}".format(len(mel_chunks)))
    return mel, wav, mel_chunks

def face_detect(video_frames, face_det_batch_size = 16, device='cuda', pads=(0,0,0,0),
        target_size = 224):
    face_detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
        flip_input=False, device=device)
    print('Detecting and cropping faces on video frames...')

    batch_size = face_det_batch_size
    n_frames = len(video_frames)

    batches = [video_frames[i:i + batch_size] for i in range(0, n_frames, batch_size)]
    i = -1
    cropped_video_frames = []
    for bacth_idx, fb in enumerate(batches):
        preds = face_detector.get_detections_for_batch(np.asarray(fb))
        preds = get_smoothened_boxes(preds, T=batch_size)
        for element_idx, bbox in enumerate(preds):
            i += 1
            if bbox is None:
                continue
            bbox = bbox.astype(np.int32)
            x1, y1, x2, y2 = bbox
            frame = fb[element_idx][y1:y2, x1:x2]
            cropped_video_frames.append(frame)
            continue
        continue
    
    assert len(cropped_video_frames) == len(video_frames)
    
    cropped_video_frames = [cv2.resize(_, (target_size, target_size)) for _ in cropped_video_frames]
    cropped_video_frames = np.array(cropped_video_frames)
    cropped_video_frames = torch.tensor(cropped_video_frames).permute(0, 3, 1, 2)
    cropped_video_frames = cropped_video_frames / 255
    return cropped_video_frames

def save_video(frames, video_path, fps=25):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width, height = frames[0].shape[:2]
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    for frame in frames:
        frame = frame.astype(np.uint8)
        out.write(frame)
    out.release()
    print(f"Annotated video saved to {video_path}")

def get_video_frames(video_path:str, resize_factor=1, rotate=False, crop = (0,-1,0,-1)):
    video_stream = cv2.VideoCapture(video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    print('Reading video frames...')

    full_frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        if resize_factor > 1:
            frame = cv2.resize(frame, (frame.shape[1]//resize_factor, frame.shape[0]//resize_factor))

        if rotate:
            frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

        y1, y2, x1, x2 = crop
        if x2 == -1: x2 = frame.shape[1]
        if y2 == -1: y2 = frame.shape[0]

        frame = frame[y1:y2, x1:x2]

        full_frames.append(frame)
    return full_frames

def main():
    # Argument Parsing
    import argparse
    parser = argparse.ArgumentParser(description='Train Lip-Sync Expert Model')
    parser.add_argument("--source_video_path", help="Path to source video file to edit", required=True)
    parser.add_argument("--target_video_path", help="Path to output file", default=None, type=str)
    parser.add_argument("--audio_path", help="Path audio to replace original", required=True)
    parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', required=True, type=str)
    parser.add_argument('--wav2lip_type', 
        help='Combination of Wav2Lip and SyncExpert backends: original==fully original, marlin_full==both use MARLIN as video encoder; marlin==SyncNet original and Wav2Lip+MARLIN', 
        default="marlin", type=str)
    
    args = parser.parse_args()
    
    source_video_path = args.source_video_path
    audio_path = args.audio_path
    video_edit_type = 'restore' if source_video_path == audio_path else 'replace'
    auto_output_name = source_video_path.split('/')[-1].replace('.', '_')+'_'+ args.checkpoint_path.split('/')[-1].replace('.ckpt', '') + '_' + video_edit_type + '.mp4'
    target_video_path = args.target_video_path if args.target_video_path is not None else auto_output_name
    
    debug_frames = True
    
    if args.wav2lip_type == 'marlin_full':
        # Params for Syncnet-Marlin and Wav2Lip-Marlin
        syncnet_T = 5
        syncnet_T_m = int(3.2 * syncnet_T)
        
        syncnet_mel_step_size = 16
        syncnet_mel_step_size_m = int(syncnet_mel_step_size * 3.18)

        # both are MARLIN-based
        syncnet_mel_step_size = syncnet_mel_step_size_m
        syncnet_T = syncnet_T_m
        
        wav2lip_type='marlin'
        hparams.img_size = 224
    elif args.wav2lip_type == 'original':
        # Params for Syncnet-orginal and Wav2Lip-original
        syncnet_T = 5
        syncnet_T_m = syncnet_T
        
        syncnet_mel_step_size = 16
        syncnet_mel_step_size_m = syncnet_mel_step_size
        
        wav2lip_type = 'original'
        hparams.img_size = 96
    elif args.wav2lip_type == 'marlin':
        # Params for Syncnet-orginal and Wav2Lip-Marlin
        syncnet_T = 5
        syncnet_mel_step_size = 16
        
        syncnet_T_m = int(3.2 * syncnet_T)       
        syncnet_mel_step_size_m = int(syncnet_mel_step_size * 3.18)
        
        wav2lip_type = 'marlin'
        hparams.img_size = 224
        
    dropout=0.05
    spectral_scale=True
    mean_context=False
        
    model = Wav2LipPLModel(hparams, syncnet_T, syncnet_mel_step_size, None, args.checkpoint_path, 
        wav2lip_type=wav2lip_type, infer_only=True, dropout=dropout, spectral_scale=spectral_scale, mean_context=mean_context)
    
    model.setup()
    model.eval()
    model.cuda()

    probe = ffmpeg.probe(source_video_path)["streams"][0]
    fps = float(probe['r_frame_rate'].split('/')[0]) / float(probe['r_frame_rate'].split('/')[1])
    hparams.set_hparam('fps', fps)
    
    video_frames = get_video_frames(video_path = source_video_path)
    cropped_video_frames = face_detect(video_frames)
    cropped_video_frames = cropped_video_frames.cuda()
    
    mel_full, wav, mel_chunks = get_mels(audio_path=audio_path, mel_step_size=syncnet_mel_step_size, fps = fps, config=hparams)
    mel_chunks = np.array(mel_chunks)
    mel_chunks = torch.tensor(mel_chunks).float().cuda()  # Ensure mel_chunks are pre-transposed
    #mel_chunks = mel_chunks / mel_chunks.abs().max()

    n_mels = 80
    
    chunk_outputs = []
    for i in range(0, mel_chunks.shape[0] - syncnet_T_m + 1, syncnet_T_m):
        # Masked and reference frames
        masked_frames = cropped_video_frames[i:i+syncnet_T_m].clone()
        reference_frames = masked_frames.clone()  
        masked_frames[:,:,masked_frames.shape[3]//2:,:] = 0.

        # Prepare input frames
        input_frames = torch.cat([masked_frames, reference_frames], dim=1).transpose(1, 0).unsqueeze(0)

        # Prepare indiv_mels
        indiv_mels = []
        for j in range(syncnet_T_m):
            mel_slice = mel_chunks[i + j]  # Transpose to match training
            if mel_slice.shape != (n_mels, syncnet_mel_step_size):  # Validate shape
                break
            indiv_mels.append(mel_slice)
        if len(indiv_mels) != syncnet_T_m:  # Skip incomplete chunks
            continue

        indiv_mels = torch.stack(indiv_mels, dim=0).unsqueeze(1).unsqueeze(0).float()  # Shape: (1, syncnet_T, 1, n_mels, syncnet_mel_step_size)

        # Model prediction
        g = model.predict_step((input_frames.cuda(), indiv_mels.cuda()), i)
        
        # Convert batch to frames 
        chunk_output = g.squeeze(0).transpose(1, 0).permute(0, 2, 3, 1).cpu()
        chunk_outputs.extend((chunk_output * 255).int().numpy())
        
        if debug_frames:
            save_batch_gifs(g, i, './output', fps, 'generated')
        continue
    
    print('Saving to: ' + target_video_path)
    
    temp_video_path = "tmp.mp4"
    save_video(chunk_outputs, temp_video_path, fps)
    #wav = wav[:int((len(chunk_outputs)+1)/fps * hparams.sample_rate)]
    temp_audio_path = save_wav(wav, hparams.sample_rate)
    combine_video_audio_ffmpeg(temp_video_path,  temp_audio_path, target_video_path)
    remove_temp_files(temp_video_path, temp_audio_path)
    return
    
    
if __name__ == "__main__":
    main()