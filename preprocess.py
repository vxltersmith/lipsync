import sys

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
	raise Exception("Must be using >= Python 3.2")

from os import listdir, path

if not path.isfile('Wav2Lip/face_detection/detection/sfd/s3fd.pth'):
	raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
							before running this script!')

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse, os, traceback, subprocess
from tqdm import tqdm
from glob import glob
import Wav2Lip.audio
from Wav2Lip.hparams import hparams as hp

import Wav2Lip.face_detection as face_detection

import cv2
import ffmpeg
import numpy as np
from typing import Union, Tuple

parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=32, type=int)
parser.add_argument("--data_root", help="Root folder of the LRS2 dataset", default='/data/CelebV-HQ/')
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", default='/data/CelebVHQ/')

args = parser.parse_args()

fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, 
									device='cuda:{}'.format(id)) for id in range(args.ngpu)]

template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'
# template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'

def rescale_bboxes(bboxes_resized, original_size, resized_size):
    """
    Rescale bounding boxes from resized image back to the original image size.

    Parameters:
    - bboxes_resized: Single bounding box or list of bounding boxes in resized image format:
                      [x_min, y_min, x_max, y_max, probability]
    - original_size: Tuple (H_orig, W_orig) of the original image size.
    - resized_size: Tuple (H_resized, W_resized) of the resized image size.

    Returns:
    - Adjusted bounding box or list of bounding boxes scaled to the original image size.
    """
    H_orig, W_orig = original_size
    H_resized, W_resized = resized_size

    # Scaling factors
    scale_x = W_orig / W_resized
    scale_y = H_orig / H_resized

    def adjust_bbox(bbox):
        # Unpack the bounding box
        x_min_resized, y_min_resized, x_max_resized, y_max_resized, prob = bbox

        # Rescale bounding box coordinates
        x_min_orig = x_min_resized * scale_x
        y_min_orig = y_min_resized * scale_y
        x_max_orig = x_max_resized * scale_x
        y_max_orig = y_max_resized * scale_y

        # Return adjusted bounding box
        return [x_min_orig, y_min_orig, x_max_orig, y_max_orig, prob]

    # Check if the input is a single bounding box or a list of bounding boxes
    if isinstance(bboxes_resized[0], (int, float)):
        # Input is a single bounding box
        return adjust_bbox(bboxes_resized)
    else:
        # Input is a list of bounding boxes
        adjusted_bboxes = []
        for bbox in bboxes_resized:
            if len(bbox) == 0:
                adjusted_bboxes.append(bbox)
            else:
                adjusted_bboxes.append(adjust_bbox(bbox))
        return adjusted_bboxes


def resize_img_cv2(images: Union[np.ndarray, list], target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Resizes a single image or a batch of images using OpenCV's cv2.resize.

    Parameters:
    - images: Either a single image (3D numpy array) or a batch of images (4D numpy array or list of images).
    - target_size: Tuple (target_height, target_width) representing the size to resize to.

    Returns:
    - Resized image or batch of resized images as a numpy array.
    """
    
    # Check if the input is a single image (3D numpy array)
    if isinstance(images, np.ndarray) and len(images.shape) == 3:
        # Resize the single image
        return cv2.resize(images, target_size)
    
    # If it's a list or 4D array, treat it as a batch of images
    resized_images = []
    for img in images:
        resized_img = cv2.resize(img, target_size)
        resized_images.append(resized_img)
    
    # Convert list of resized images to numpy array with shape (batch_size, target_height, target_width, channels)
    return np.array(resized_images)

def process_video_file(vfile, args, gpu_id, check_before_skip = False, max_diff_number=10, max_frame_size = (512,512)):
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]
    fulldir = path.join(args.preprocessed_root, dirname, vidname)

    probe = ffmpeg.probe(vfile)["streams"][0]
    n_frames = int(probe["nb_frames"])
    orig_size = (probe['height'], probe['width'])

    if os.path.exists(fulldir):
        if not check_before_skip:
            return
        files = os.listdir(fulldir)
        if n_frames - len(files) < max_diff_number:
            return
    
    video_stream = cv2.VideoCapture(vfile)
	
    frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frame_size = frame.shape[:2]
        if frame_size > max_frame_size:
            frame = resize_img_cv2(frame, max_frame_size)
        frames.append(frame)
 
    os.makedirs(fulldir, exist_ok=True)
  
    batches = [frames[i:i + args.batch_size] for i in range(0, n_frames, args.batch_size)]

    i = -1
    for fb in batches:
        preds = fa[gpu_id].get_detections_for_batch(np.asarray(fb))
        for j, f in enumerate(preds):
            i += 1
            if f is None:
                continue
            x1, y1, x2, y2 = f
            cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), fb[j][y1:y2, x1:x2])
            continue
        continue
    return

def process_audio_file(vfile, args):
	vidname = os.path.basename(vfile).split('.')[0]
	dirname = vfile.split('/')[-2]

	fulldir = path.join(args.preprocessed_root, dirname, vidname)
	os.makedirs(fulldir, exist_ok=True)

	wavpath = path.join(fulldir, 'audio.wav')

	command = template.format(vfile, wavpath)
	subprocess.call(command, shell=True)

	
def mp_handler(job):
    vfile, args, gpu_id = job
    #process_video_file(vfile, args, gpu_id)
    try:
        process_video_file(vfile, args, gpu_id)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()
		
def main(args):
	print('Started processing for {} with {} GPUs'.format(args.data_root, args.ngpu))

	filelist = glob(path.join(args.data_root, '*/*.mp4'))

	jobs = [(vfile, args, i%args.ngpu) for i, vfile in enumerate(filelist)]
	p = ThreadPoolExecutor(args.ngpu)
	futures = [p.submit(mp_handler, j) for j in jobs]
	_ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

	print('Dumping audios...')

	for vfile in tqdm(filelist):
		try:
			process_audio_file(vfile, args)
		except KeyboardInterrupt:
			exit(0)
		except:
			traceback.print_exc()
			continue

if __name__ == '__main__':
	main(args)