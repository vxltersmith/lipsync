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
import tqdm
import os

from realesrgan_proxy import proxy_call_realesrgan


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
    return wav, mel_chunks

def get_wav_chunks(audio_path: str, frames_step_size: int, fps: int, config, overlapping: bool = False):
    print('Read audio and slice wavs...')
    import Wav2LipMarlin.datasets.audio as audio
    wav = audio.load_wav(audio_path, config.sample_rate)
    
    wav_chunks = []
    wav_idx_multiplier = int(frames_step_size/fps * config.sample_rate) if not overlapping else int(1/fps * config.sample_rate)
    wav_step_size = int(frames_step_size/fps * config.sample_rate)
    i = 0
    while 1:
        start_idx = int(i * wav_idx_multiplier)
        if start_idx + wav_step_size > len(wav):
            last_chunk = wav[start_idx:-1]
            full_last_chunk = np.zeros((wav_step_size))
            full_last_chunk[:last_chunk.shape[-1]] = last_chunk
            wav_chunks.append(full_last_chunk)
            break
        wav_chunks.append(wav[start_idx : start_idx + wav_step_size])
        i += 1
    print("Length of wav chunks: {}".format(len(wav_chunks)))
    return wav, wav_chunks

def face_detect(video_frames, face_det_batch_size = 16, device='cuda', pads=(0,0,0,0),
        target_size = 224, final_face_resolution=96, resize_to_gen = True):
    face_detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
        flip_input=False, device=device)
    print('Detecting and cropping faces on video frames...')

    batch_size = face_det_batch_size
    n_frames = len(video_frames)

    batches = [video_frames[i:i + batch_size] for i in range(0, n_frames, batch_size)]
    i = -1
    
    cropped_video_frames = []
    resized_frames = []  # To store resized whole frames
    face_positions = []  # To store positions of faces in the original frame
    for bacth_idx, fb in tqdm.tqdm(enumerate(batches)):
        preds = face_detector.get_detections_for_batch(np.asarray(fb))

        if any(_ is None for _ in preds):
            print(f'face wasnt detected in batch {bacth_idx}. Skipping rest of the video...')
            break

        preds = get_smoothened_boxes(preds, T=batch_size)
        for element_idx, bbox in enumerate(preds):
            i += 1
            if bbox is None:
                continue
            bbox = bbox.astype(np.int32)
            x1, y1, x2, y2 = bbox
            
            frame = fb[element_idx].copy()
            frame_h, frame_w = frame.shape[:2]
            cropped_frame = fb[element_idx][y1:y2, x1:x2].copy()
            cropped_video_frames.append(cropped_frame)
            #frame[y1:y2, x1:x2] = 125
            
            if resize_to_gen:
                # Calculate scaling factor based on the face resolution
                # Resize the full frame to maintain its aspect ratio
                frame_aspect_ratio = frame_h / frame_w
                if frame_aspect_ratio > 1:
                    # Taller frame
                    bbox_h = y2-y1             
                    new_height = int(final_face_resolution * frame_h / bbox_h)
                    new_width = int(new_height / frame_aspect_ratio)
                else:
                    # Wider frame
                    bbox_w = x2-x1
                    new_width = int(final_face_resolution * frame_w / bbox_w)
                    new_height = int(new_width * frame_aspect_ratio)

                resized_frame = cv2.resize(frame, (new_width, new_height))
            else:
                resized_frame = frame
            resized_frames.append(resized_frame)

            if resize_to_gen:
                # Calculate scaled face bounding box for the resized frame
                scale_x = new_width / frame_w
                scale_y = new_height / frame_h
                scaled_bbox = [
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y)
                ]
            else:
                scaled_bbox = [x1, y1, x2, y2]
            face_positions.append(scaled_bbox)
            continue
        continue
    
    assert len(face_positions) == len(cropped_video_frames) == len(resized_frames)
    
    cropped_video_frames = [cv2.resize(_, (target_size, target_size)) for _ in cropped_video_frames]
    cropped_video_frames = np.array(cropped_video_frames)
    cropped_video_frames = torch.tensor(cropped_video_frames).permute(0, 3, 1, 2)
    cropped_video_frames = cropped_video_frames / 255
    return cropped_video_frames, face_positions, resized_frames

def save_video(frames, video_path, fps=25):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width, height = frames[0].shape[:2]
    out = cv2.VideoWriter(video_path, fourcc, fps, (height, width))
    for frame in frames:
        if frame.shape[:2] != (width, height):
            frame = cv2.resize(frame, (height, width))
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

def main(source_video_path, audio_path, target_video_path, audio_encoder_type, start_at, output_dir, 
        wav2lip_type, checkpoint_path, gif_dir, model_args,
        debug_frames = False, embed_full_frame = False):
    # make name for video output
    video_edit_type = 'restore' if source_video_path == audio_path else 'replace'
    auto_output_name = '__'.join(
        [os.path.basename(audio_path).split('.')[0], 
         os.path.basename(source_video_path).split('.')[0], 
         os.path.basename(checkpoint_path).split('.')[0], 
         video_edit_type + '.mp4']
    )
    target_video_path = target_video_path if target_video_path is not None else auto_output_name
    
    os.makedirs(output_dir, exist_ok=True)
    gif_dir = os.path.join(output_dir, gif_dir)
    target_video_path = os.path.join(output_dir, target_video_path)
    
    if wav2lip_type == 'marlin_full':
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
    elif wav2lip_type == 'original':
        # Params for Syncnet-orginal and Wav2Lip-original
        syncnet_T = 5
        syncnet_T_m = syncnet_T
        
        syncnet_mel_step_size = 16
        syncnet_mel_step_size_m = syncnet_mel_step_size
        
        wav2lip_type = 'original'
        hparams.img_size = 96
    elif wav2lip_type == 'marlin':
        # Params for Syncnet-orginal and Wav2Lip-Marlin
        syncnet_T = 5
        syncnet_mel_step_size = 16
        
        syncnet_T_m = int(3.2 * syncnet_T)       
        syncnet_mel_step_size_m = int(syncnet_mel_step_size * 3.18)
        
        wav2lip_type = 'marlin'
        hparams.img_size = 224
        
    dropout = model_args['dropout']
    spectral_scale = model_args['spectral_scale']
    mean_context = model_args['mean_context']
    resize_to_gen = model_args['resize_to_gen']
    deep_fusion = model_args['deep_fusion']
    overlapping = model_args.get('overlapping', True)
        
    model = Wav2LipPLModel(hparams, syncnet_T, syncnet_mel_step_size, None, checkpoint_path, 
        wav2lip_type=wav2lip_type, infer_only=True, dropout=dropout, spectral_scale=spectral_scale, 
        mean_context=mean_context, audio_encoder_type = audio_encoder_type, deep_fusion = deep_fusion)
    
    model.setup()
    torch.compile(model.model)
    model.eval()
    model.cuda()

    probe = ffmpeg.probe(source_video_path)["streams"][0]
    fps = float(probe['r_frame_rate'].split('/')[0]) / float(probe['r_frame_rate'].split('/')[1])
    hparams.set_hparam('fps', fps)
    start_at = int(start_at*fps)
    
    original = audio_encoder_type == 'default'
    if original:
        wav, mel_chunks = get_mels(audio_path=audio_path, mel_step_size=syncnet_mel_step_size, fps = fps, config=hparams)
        mel_chunks = np.array(mel_chunks)
        mel_chunks = torch.tensor(mel_chunks).float().cpu()  # Ensure mel_chunks are pre-transposed
        n_mels = 80
        total_audio_chunks = mel_chunks.shape[0]
        inference_steps = total_audio_chunks - syncnet_T_m
        step = syncnet_T_m
    else:
        wav, wav_chunks = get_wav_chunks(audio_path=audio_path, frames_step_size=syncnet_T_m, fps = fps,
            config=hparams, overlapping=overlapping)
        wav_chunks = np.array(wav_chunks)
        wav_chunks = torch.tensor(wav_chunks).float().cpu()  # Ensure mel_chunks are pre-transposed
        total_audio_chunks = wav_chunks.shape[0]
        inference_steps = total_audio_chunks
        step = 1
        
    video_frames = get_video_frames(video_path = source_video_path)
    if start_at > len(video_frames)-1:
        start_at = 0
    end_at = min(int(wav.shape[-1]/hparams.sample_rate*fps), len(video_frames)-1) + start_at
    video_frames = video_frames[start_at:end_at]
    
    # if not original:
    #     assert len(video_frames) == len(wav_chunks)
    
    cropped_video_frames, face_positions, resized_frames = face_detect(video_frames, resize_to_gen=resize_to_gen)
    cropped_video_frames = cropped_video_frames.cpu()
    
    # For videos shorter than audio
    video_frames_num = int(cropped_video_frames.shape[0])
    sinthes_steps = total_audio_chunks/step
    required_frames = int(sinthes_steps * syncnet_T_m)
    if required_frames > video_frames_num:
        factor = (required_frames // video_frames_num) + 1
        cropped_video_frames = torch.cat([cropped_video_frames.clone() for _ in range(factor)], dim=0)[:required_frames]
        resized_frames = (resized_frames * factor)[:required_frames]
        face_positions = (face_positions * factor)[:required_frames]
        
    synced_resized_frames = []
    synced_resized_bboxes = []
    
    assert sinthes_steps * syncnet_T_m <= len(cropped_video_frames)

    chunk_outputs = []
    for i in tqdm.tqdm(range(0, inference_steps, step)):
        if original:
            start = i
        else:
            start = max(0, i*syncnet_T_m-1) if not overlapping else i
        end = start+syncnet_T_m
        
        if end > len(cropped_video_frames):
            end_diff = min(end-len(cropped_video_frames)+1, syncnet_T_m)
            end = -1
        else:
            end_diff = None
            
        # Masked and reference frames
        masked_frames = cropped_video_frames[start:end].clone()
        # In case of length mismatch on chunk edges
        if end_diff is not None:
            additional_frames = cropped_video_frames[-end_diff:].clone()
            masked_frames = torch.cat([masked_frames, additional_frames], dim=0)
        
        #reference_frames =  cropped_video_frames[0:syncnet_T_m].clone()
        #masked_frames = reference_frames.clone()
        reference_frames = torch.flip(masked_frames.clone(), [0])
        # T, C, H, W = masked_frames.shape
        # random_frame = masked_frames[0,:, :,:].clone().unsqueeze(0)  # Shape (B, C, 1, H, W)
        # reference_frames =  random_frame.expand(T, C, H, W)  # Expand to (T, C, H, W)
        masked_frames[:,:,masked_frames.shape[3]//2:,:] = 0.

        # Prepare input frames
        input_frames = torch.cat([masked_frames, reference_frames], dim=1).transpose(1, 0).unsqueeze(0)

        if original:
            # Prepare indiv_mels
            indiv_mels = []
            for j in range(syncnet_T_m):
                mel_slice = mel_chunks[i + j]
                if mel_slice.shape != (n_mels, syncnet_mel_step_size):  # Validate shape
                    break
                indiv_mels.append(mel_slice)
            if len(indiv_mels) != syncnet_T_m:  # Skip incomplete chunks
                continue
            indiv_mels = torch.stack(indiv_mels, dim=0).unsqueeze(1).unsqueeze(0).float()  # Shape: (1, syncnet_T, 1, n_mels, syncnet_mel_step_size)
        else:
            indiv_mels = wav_chunks[i]
            indiv_mels = indiv_mels.unsqueeze(0)

        # Model prediction
        g = model.predict_step((input_frames.cuda(), indiv_mels.cuda()), i)
        
        middle_id = 8
        first = i == 0
        last = i == inference_steps - middle_id
        first_or_last = first or last
        
        # Convert batch to frames 
        chunk_output = g.squeeze(0).transpose(1, 0).permute(0, 2, 3, 1).cpu()
        chunk_output = (chunk_output * 255).int().numpy()
        
        if overlapping:
            if not first_or_last:
                chunk_outputs.append(chunk_output[middle_id].copy())
                synced_resized_frames.append(resized_frames[start+middle_id].copy())
                synced_resized_bboxes.append(face_positions[start+middle_id].copy())
                assert len(chunk_outputs) == len(synced_resized_frames)
                continue
            elif first:
                chunk_output = chunk_output[:middle_id]
                resized_frames_middle = resized_frames[:middle_id]
                face_positions_middle = face_positions[:middle_id]
            elif last:
                chunk_output = chunk_output[:middle_id]
                resized_frames_middle = resized_frames[start:start+middle_id]
                face_positions_middle = face_positions[start:start+middle_id]                
                
            chunk_outputs.extend([_.copy() for _ in chunk_output])
            # For later we need to copy exact same initial frames and BBoxes
            synced_resized_frames.extend([_.copy() for _ in resized_frames_middle])
            synced_resized_bboxes.extend([_.copy() for _ in face_positions_middle])
            assert len(chunk_outputs) == len(synced_resized_frames)
            
            if last: break
            continue
            
        chunk_outputs.extend(chunk_output)
        # For later we need to copy exact same initial frames and BBoxes
        synced_resized_frames.extend([_.copy() for _ in resized_frames[start:end]])
        synced_resized_bboxes.extend([_.copy() for _ in face_positions[start:end]])
        # In case of length mismatch on chunk edges
        if end_diff is not None:
            synced_resized_frames.extend([_.copy() for _ in resized_frames[-end_diff:]])
            synced_resized_bboxes.extend([_.copy() for _ in face_positions[-end_diff:]])
        continue
    
    if debug_frames: # dump is gif
        save_batch_gifs([chunk_outputs], i, gif_dir, fps, 'generated', raw=False)
    
    # Embedding faces back to resized original frames
    smoothed_bboxes = []
    previous_bbox = None
    for i, bbox in enumerate(synced_resized_bboxes):
        bbox = [int(_) for _ in bbox]
        smoothed_bboxes.append(bbox)
    
    assert len(chunk_outputs) == len(synced_resized_frames)
    
    resized_frames_with_faces = []
    for i, resized_frame in enumerate(synced_resized_frames):
        # Extract bounding box for embedding
        x1, y1, x2, y2 = smoothed_bboxes[i]
        if i > len(chunk_outputs) - 1:
            break
        resized_face = chunk_outputs[i]
        bbox_w = int(x2-x1)
        bbox_h = int(y2-y1)
        resized_face = torch.nn.functional.interpolate(
            torch.tensor(resized_face).permute(2, 0, 1).unsqueeze(0).float(), (bbox_h, bbox_w)).squeeze().permute(1, 2, 0).int()
        if embed_full_frame:
            resized_frame[y1:y2, x1:x2] = resized_face  # Embed the resized face back
        else:
            resized_face = resized_face[bbox_h//2:,:]
            lover_shape = resized_frame[y1+bbox_h//2:y2, x1:x2].shape
            resized_face = torch.nn.functional.interpolate(
                torch.tensor(resized_face).permute(2, 0, 1).unsqueeze(0).float(), (lover_shape[0], lover_shape[1])).squeeze().permute(1, 2, 0).int()
            resized_frame[y1+bbox_h//2:y2, x1:x2] = resized_face
        resized_frames_with_faces.append(resized_frame)
    
    print('Saving video...')
    temp_video_path = "tmp.mp4"
    save_video(resized_frames_with_faces, temp_video_path, fps)
    #wav = wav[:int((len(chunk_outputs)+1)/fps * hparams.sample_rate)]
    temp_audio_path = save_wav(wav, hparams.sample_rate)
    combine_video_audio_ffmpeg(temp_video_path,  temp_audio_path, target_video_path)
    remove_temp_files(temp_video_path, temp_audio_path)
    print('Saved to: ' + target_video_path)
    return target_video_path
    
    
if __name__ == "__main__":
        # Argument Parsing
    import argparse
    parser = argparse.ArgumentParser(description='Train Lip-Sync Expert Model')
    parser.add_argument("--source_video_path", help="Path to source video file to edit", required=True)
    parser.add_argument("--target_video_path", help="Path to output file", default=None, type=str)
    parser.add_argument("--gif_dir", help="Path to output dir with gifs", default='gifs/', type=str)
    parser.add_argument("--output_dir", help="Path to output dir with gifs", default='./wav2vec_marlin_results/', type=str)
    parser.add_argument("--audio_path", help="Path audio to replace original", required=True)
    parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', required=True, type=str)
    parser.add_argument('--start_at', help='Use reference video from specified second', default=0, type=int)
    parser.add_argument('--audio_encoder_type', help='How to exctract audio features', default="default", type=str)
    parser.add_argument('--wav2lip_type', 
        help='Combination of Wav2Lip and SyncExpert backends: original==fully original, marlin_full==both use MARLIN as video encoder; marlin==SyncNet original and Wav2Lip+MARLIN', 
        default="marlin", type=str)
    
    args = parser.parse_args()
    
    source_video_path = args.source_video_path
    audio_path = args.audio_path
    target_video_path = args.target_video_path
    gif_dir = args.gif_dir
    output_dir = args.output_dir
    checkpoint_path = args.checkpoint_path
    start_at = args.start_at
    audio_encoder_type = args.audio_encoder_type
    wav2lip_type = args.wav2lip_type
    
    model_args = {
        'dropout': 0.05,
        'spectral_scale': True,
        'mean_context': False,
        'resize_to_gen': True,
        'deep_fusion': True,
        'overlapping': False
    }
    
    target_video_path = main(source_video_path, audio_path=audio_path, target_video_path=target_video_path, audio_encoder_type=audio_encoder_type, 
        start_at=start_at, output_dir=output_dir, wav2lip_type=wav2lip_type, checkpoint_path=checkpoint_path, gif_dir=gif_dir,
        model_args = model_args
    )
    
    proxy_call_realesrgan(
        input_path=target_video_path,
        output_path=os.path.join(output_dir, 'output_videos_hd'),
        model_name='RealESRGAN_x2plus',
        outscale=2,
        face_enhance=True,
    )
    