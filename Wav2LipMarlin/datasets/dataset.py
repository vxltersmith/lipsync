from glob import glob
from os.path import dirname, join, basename, isfile
import Wav2LipMarlin.datasets.audio as audio
import os, random
import cv2
import torch
import numpy as np


class Wav2LipDataset(object):
    def __init__(self, data_root, split, hparams, syncnet_T, syncnet_mel_step_size):
        def get_image_list(data_root, split):
            filelist = []

            with open('filelists/{}.txt'.format(split)) as f:
                for line in f:
                    line = line.strip()
                    if ' ' in line: line = line.split()[0]
                    filelist.append(os.path.join(data_root, line))

            return filelist
        
        self.hparams = hparams
        self.syncnet_T = syncnet_T
        self.syncnet_mel_step_size = syncnet_mel_step_size
        self.all_videos = get_image_list(data_root, split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + self.syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(self.hparams.fps)))

        end_idx = start_idx + self.syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]

            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * self.syncnet_T:
                continue

            img_name = random.choice(img_names)
            if random.choice([True, False]):
                y = torch.ones(1).float()
                chosen = img_name
            else:
                y = torch.zeros(1).float()
                wrong_img_name = random.choice(img_names)
                while wrong_img_name == img_name:
                    wrong_img_name = random.choice(img_names)
                chosen = wrong_img_name

            window_fnames = self.get_window(chosen)
            if window_fnames is None:
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (self.hparams.img_size, self.hparams.img_size))
                except Exception as e:
                    all_read = False
                    break

                window.append(img)

            if not all_read: continue

            #try:
            wavpath = join(vidname, "audio.wav")
            wav = audio.load_wav(wavpath, self.hparams.sample_rate)

            orig_mel = audio.melspectrogram(wav, self.hparams).T
            #except Exception as e:
            #    continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != self.syncnet_mel_step_size):
                continue

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1]//2:]

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return x, mel, y
        
from Wav2LipMarlin.datasets.video import read_video_as_numpy
class FullVideoSyncDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split, hparams, syncnet_T, syncnet_mel_step_size, wrong_only, offset = 16):
        def get_image_list(data_root, split):
            filelist = []

            with open('filelists/{}.txt'.format(split)) as f:
                for line in f:
                    line = line.strip()
                    if ' ' in line: line = line.split()[0]
                    filelist.append(os.path.join(data_root, line))

            return filelist
        
        self.hparams = hparams
        self.syncnet_T = syncnet_T
        self.syncnet_mel_step_size = syncnet_mel_step_size
        self.all_videos = get_image_list(data_root, split)
        self.wrong_only = wrong_only
        self.offset = offset

    def __len__(self):
        return len(self.all_videos)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])    
    
    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(self.hparams.fps)))
        end_idx = -1
        return spec[start_idx : end_idx, :]
    
    def get_window(self, start_frame, last_frame_id):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, last_frame_id):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame) or not os.path.exists(frame):
                return None
            window_fnames.append(frame)
        
        if len(window_fnames) == 0: return None
        
        return window_fnames

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]

            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= self.syncnet_T:
                continue
            #get sorted names of files
            sorted_available_ids = sorted([int(_.split('.')[0].split('/')[-1]) for _ in img_names])
            sorted_files = [str(_)+'.jpg' for _ in sorted_available_ids]
            img_names = [join(vidname, _) for _ in sorted_files]

            img_name = img_names[0]
            last_frame_id = sorted_available_ids[-1]
            
            if not self.wrong_only:
                y = torch.ones(1).float()
                window_fnames = self.get_window(img_name, last_frame_id)
            else:
                y = torch.zeros(1).float()
                wrong_img_name = img_names[self.offset]
                window_fnames = self.get_window(wrong_img_name, last_frame_id)
                
            if window_fnames is None:
                continue
                
            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (self.hparams.img_size, self.hparams.img_size))
                except Exception as e:
                    all_read = False
                    break
                window.append(img)

            if not all_read: continue
            
            #try:
            wavpath = join(vidname, "audio.wav")
            wav = audio.load_wav(wavpath, self.hparams.sample_rate)
            orig_mel = audio.melspectrogram(wav, self.hparams).T
            #except Exception as e:
            #    continue
            if self.wrong_only:
                mel = self.crop_audio_window(orig_mel.copy(), img_name)
                # if (mel.shape[0] != self.syncnet_mel_step_size):
                #     continue
            else:
                mel = orig_mel
            
            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1]//2:]
            
            # form a bacth
            #x = np.stack(window, axis=0)
            x = x[:(x.shape[0]//3//self.syncnet_T * self.syncnet_T) * 3,:,:]
            x = x.reshape(-1, self.syncnet_T, 3, x.shape[-2], x.shape[-1])
            # reshape to (B, C, T, H, W)
            x = x.transpose(0,2,1,3,4)
            
            mel = mel[:(mel.shape[0]//self.syncnet_mel_step_size) * self.syncnet_mel_step_size,:]
            mel = mel.reshape(-1, self.syncnet_mel_step_size, mel.shape[-1])
            mel = mel[:x.shape[0], :, :]
            mel = mel.transpose(0, 2, 1)

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel).unsqueeze(1) # should be B, 1, 80, T
            
            if x.shape[0] == 0:
                continue
            
            if x.shape[0] > mel.shape[0]:
                x = x[:mel.shape[0]]
                #continue
            

            return x, mel, y