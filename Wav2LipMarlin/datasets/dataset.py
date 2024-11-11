from glob import glob
from os.path import dirname, join, basename, isfile
import Wav2LipMarlin.datasets.audio as audio
import os, random
import cv2
import torch
import numpy as np
import ffmpeg
import torchvision.transforms as transforms



class Wav2LipDataset(object):
    def __init__(self,
        data_root,
        split,
        hparams,
        syncnet_T,
        syncnet_mel_step_size,
        syncnet_T_orig=5,
        manifest_root="filelists",
    ):
        def get_image_list(data_root, split, manifest_root):
            filelist = []
            with open(os.path.join(manifest_root, "{}.txt".format(split))) as f:
                for line in f:
                    line = line.strip()
                    if " " in line:
                        line = line.split()[0]
                    filelist.append(os.path.join(data_root, line))
            return filelist

        self.hparams = hparams
        self.syncnet_T = syncnet_T
        self.syncnet_T_orig = syncnet_T_orig
        self.max_multiplier = {5: 3, 16: 2}
        self.syncnet_mel_step_size = syncnet_mel_step_size
        self.all_videos = get_image_list(data_root, split, manifest_root)

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

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (self.hparams.img_size, self.hparams.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame, fps: float):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame) # 0-indexing ---> 1-indexing
        start_idx = int(80. * (start_frame_num / float(fps)))
        
        end_idx = start_idx + self.syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame, fps: float):
        mels = []
        assert self.syncnet_T == 5 or self.syncnet_T == 16 
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + self.syncnet_T):
            m = self.crop_audio_window(spec, i - 2, fps)
            if m.shape[0] != self.syncnet_mel_step_size:
                return None
            mels.append(m.T)

        assert len(mels) == self.syncnet_T
        
        mels = np.asarray(mels)
        return mels # Frames, n_mels, T

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))
        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= self.max_multiplier[self.syncnet_T] * self.syncnet_T:
                continue
            
            video_filename = os.path.basename(vidname) + ".mp4"
            video_path = os.path.join(vidname, video_filename)
            if os.path.exists(video_path):
                probe = ffmpeg.probe(video_path)["streams"][0] 
                fps = float(probe['r_frame_rate'].split('/')[0]) / float(probe['r_frame_rate'].split('/')[1])
            else:
                fps = self.hparams.fps
            
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue

            window = self.read_window(window_fnames)
            if window is None:
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                continue

            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, self.hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav, self.hparams).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name, fps)
            
            if (mel.shape[0] != self.syncnet_mel_step_size):
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name, fps)
            if indiv_mels is None: continue

            window = self.prepare_window(window)
            y = window.copy()
            window[:, :, window.shape[2]//2:] = 0.

            wrong_window = self.prepare_window(wrong_window)
            x = np.concatenate([window, wrong_window], axis=0)

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            y = torch.FloatTensor(y)
            return x, indiv_mels, mel, y

class Wav2LipSyncNetDataset(object):
    def __init__(
        self,
        data_root,
        split,
        hparams,
        syncnet_T,
        syncnet_mel_step_size,
        manifest_root="filelists",
    ):
        def get_image_list(data_root, split, manifest_root):
            filelist = []

            with open(os.path.join(manifest_root, "{}.txt".format(split))) as f:
                for line in f:
                    line = line.strip()
                    if " " in line:
                        line = line.split()[0]
                    filelist.append(os.path.join(data_root, line))

            return filelist

        self.hparams = hparams
        self.syncnet_T = syncnet_T
        self.syncnet_mel_step_size = syncnet_mel_step_size
        self.all_videos = get_image_list(data_root, split, manifest_root)

    def get_frame_id(self, frame):
        return int(basename(frame).split(".")[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + self.syncnet_T):
            frame = join(vidname, "{}.jpg".format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80.0 * (start_frame_num / float(self.hparams.fps)))

        end_idx = start_idx + self.syncnet_mel_step_size

        return spec[start_idx:end_idx, :]

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]

            img_names = list(glob(join(vidname, "*.jpg")))
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
                    img = cv2.resize(
                        img, (self.hparams.img_size, self.hparams.img_size)
                    )
                except Exception as e:
                    all_read = False
                    break

                window.append(img)

            if not all_read:
                continue

            # try:
            wavpath = join(vidname, "audio.wav")
            wav = audio.load_wav(wavpath, self.hparams.sample_rate)

            orig_mel = audio.melspectrogram(wav, self.hparams).T
            # except Exception as e:
            #    continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if mel.shape[0] != self.syncnet_mel_step_size:
                continue

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.0
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1] // 2 :]

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return x, mel, y


class FasterWav2LipSyncNetDataset(object):
    def __init__(
        self,
        data_root,
        split,
        hparams,
        syncnet_T,
        syncnet_mel_step_size,
        manifest_root="filelists",
    ):
        def get_image_list(data_root, split, manifest_root):
            filelist = []
            with open(os.path.join(manifest_root, "{}.txt".format(split))) as f:
                for line in f:
                    line = line.strip()
                    if " " in line:
                        line = line.split()[0]
                    filelist.append(os.path.join(data_root, line))
            return filelist

        self.hparams = hparams
        self.syncnet_T = syncnet_T
        self.syncnet_mel_step_size = syncnet_mel_step_size
        self.all_videos = get_image_list(data_root, split, manifest_root)
        print(f"Reading dataset from manifest {split} in {manifest_root}")
        self.read_and_map()

    def read_and_map(self):
        import tqdm

        all_videos = self.all_videos
        syncnet_T = self.syncnet_T

        short_video_idx = []
        extremelly_short_idx = []
        long_videos_idx = []
        medium_videos_idx = []

        def get_sorted_names(vidname, img_names):
            # get sorted names of files
            sorted_available_ids = sorted(
                [int(_.split(".")[0].split("/")[-1]) for _ in img_names]
            )
            sorted_files = [str(_) + ".jpg" for _ in sorted_available_ids]
            img_names = [join(vidname, _) for _ in sorted_files]
            return img_names, sorted_available_ids

        all_videos_mapping = {}
        for idx in tqdm.tqdm(range(len(all_videos))):
            vidname = all_videos[idx]
            img_names = list(glob(join(vidname, "*.jpg")))

            video_len_type = None
            if len(img_names) < syncnet_T:
                extremelly_short_idx.append(idx)
                video_len_type = "ext_short"
                # exclude from dataset
                continue
            if len(img_names) <= 2 * syncnet_T:
                short_video_idx.append(idx)
                video_len_type = "short"  # if short replace audio
            elif len(img_names) >= 3 * syncnet_T:
                long_videos_idx.append(idx)
                video_len_type = "long"  # default logic
            else:
                medium_videos_idx.append(idx)
                video_len_type = "medium"  # fixed offset

            sorted_img_names, sorted_available_ids = get_sorted_names(
                vidname, img_names
            )
            all_videos_mapping[idx] = (
                vidname,
                video_len_type,
                sorted_img_names,
                sorted_available_ids,
            )
            continue
        self.all_videos_mapping = all_videos_mapping
        self.video_with_long_audios = medium_videos_idx + long_videos_idx

    def get_frame_id(self, frame):
        return int(basename(frame).split(".")[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + self.syncnet_T):
            frame = join(vidname, "{}.jpg".format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80.0 * (start_frame_num / float(self.hparams.fps)))

        end_idx = start_idx + self.syncnet_mel_step_size

        return spec[start_idx:end_idx, :]

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.choice(list(self.all_videos_mapping.keys()))
            vidname, video_len_type, sorted_img_names, sorted_available_ids = (
                self.all_videos_mapping[idx]
            )

            img_names = sorted_img_names

            if random.choice([True, False]):
                y = torch.ones(1).float()  # if synced we take 0st frame
                if video_len_type == "medium" or video_len_type == "long":
                    chosen = random.choice(img_names[: -self.syncnet_T])
                else:
                    start_idx = random.randint(0, max(0, len(img_names) - self.syncnet_T))
                    chosen = img_names[start_idx]
                crop_audio_from_frame = chosen
            else:
                y = torch.zeros(1).float()
                crop_audio_from_frame_idx = sorted_available_ids[0]
                if (
                    video_len_type == "short"
                ):  # need to replace audio, taking first frame of video
                    chosen = img_names[0]
                else:
                    wrong_img_idx = random.choice(
                        sorted_available_ids[
                            0 : max(1, len(sorted_available_ids) - self.syncnet_T)
                        ]
                    )
                    try:
                        chosen = img_names[wrong_img_idx]
                    except:
                        chosen = img_names[0]

                    # if video_len_type == 'long':
                    left_distance = len(sorted_available_ids[:wrong_img_idx])
                    right_distance = len(
                        sorted_available_ids[wrong_img_idx : -self.syncnet_T]
                    )
                    offset = self.hparams.fps // 5
                    if right_distance > left_distance:
                        crop_audio_from_available_idx = sorted_available_ids[
                            wrong_img_idx + offset : -self.syncnet_T
                        ]
                    else:
                        crop_audio_from_available_idx = sorted_available_ids[
                            0 : wrong_img_idx - offset
                        ]

                    if crop_audio_from_available_idx:
                        crop_audio_from_frame_idx = random.choice(
                            crop_audio_from_available_idx
                        )
                    else:
                        crop_audio_from_frame_idx = sorted_available_ids[
                            0
                        ]  # or a fallback index

                if crop_audio_from_frame_idx >= len(img_names):
                    print("Warning: crop_audio_from_frame_idx out of bounds.")
                    crop_audio_from_frame_idx = sorted_available_ids[0]
                crop_audio_from_frame = img_names[crop_audio_from_frame_idx]

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
                    img = cv2.resize(
                        img, (self.hparams.img_size, self.hparams.img_size)
                    )
                except Exception as e:
                    all_read = False
                    break

                window.append(img)

            if not all_read:
                continue

            # get mel for clip
            wavpath = join(vidname, "audio.wav")
            if (
                y.item() == 0 and video_len_type == "short"
            ):  # need to replace audio path
                new_audio_idx = random.choice(self.video_with_long_audios)
                new_vidname, _, _, _ = self.all_videos_mapping[new_audio_idx]
                wavpath = join(new_vidname, "audio.wav")

            wav = audio.load_wav(wavpath, self.hparams.sample_rate)
            orig_mel = audio.melspectrogram(wav, self.hparams).T

            mel = self.crop_audio_window(orig_mel.copy(), crop_audio_from_frame)

            if mel.shape[0] != self.syncnet_mel_step_size:
                continue

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.0
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1] // 2 :]

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return x, mel, y


class FullVideoSyncDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        split,
        hparams,
        syncnet_T,
        syncnet_mel_step_size,
        wrong_only,
        offset=16,
        rand_idx=False,
    ):
        def get_image_list(data_root, split):
            filelist = []
            with open("filelists/{}.txt".format(split)) as f:
                for line in f:
                    line = line.strip()
                    if " " in line:
                        line = line.split()[0]
                    filelist.append(os.path.join(data_root, line))
            return filelist

        self.hparams = hparams
        self.syncnet_T = syncnet_T
        self.syncnet_mel_step_size = syncnet_mel_step_size
        self.all_videos = get_image_list(data_root, split)
        self.data_root = data_root
        self.wrong_only = wrong_only
        self.offset = offset
        self.save_for_debug = False
        self.rand_idx = rand_idx

    def __len__(self):
        return len(self.all_videos)

    def get_frame_id(self, frame):
        return int(basename(frame).split(".")[0])

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80.0 * (start_frame_num / float(self.hparams.fps)))
        end_idx = -1
        return spec[start_idx:end_idx, :]

    def get_window(self, start_frame, last_frame_id):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, last_frame_id):
            frame = join(vidname, "{}.jpg".format(frame_id))
            if not isfile(frame) or not os.path.exists(frame):
                return None
            window_fnames.append(frame)

        if len(window_fnames) == 0:
            return None

        return window_fnames

    def random_indx(self):
        return random.randint(0, len(self.all_videos) - 1)

    def empty_output(self):
        return torch.zeros(1, 1, 1), torch.zeros(1, 1, 1), torch.zeros(1)

    def __getitem__(self, idx):
        # to do - remove randomnes
        vidname = self.all_videos[idx]

        img_names = list(glob(join(vidname, "*.jpg")))
        if len(img_names) <= self.syncnet_T:
            return self.empty_output()
        # get sorted names of files
        sorted_available_ids = sorted(
            [int(_.split(".")[0].split("/")[-1]) for _ in img_names]
        )
        sorted_files = [str(_) + ".jpg" for _ in sorted_available_ids]
        img_names = [join(vidname, _) for _ in sorted_files]

        img_name = img_names[0]
        last_frame_id = sorted_available_ids[-1]

        if not self.wrong_only:
            y = torch.ones(1).float()
            window_fnames = self.get_window(img_name, last_frame_id)
        else:
            y = torch.zeros(1).float()
            misaligned_index = min(self.offset, len(img_names) - 1)  # Ensure index is within bounds
            wrong_img_name = img_names[misaligned_index]
            window_fnames = self.get_window(wrong_img_name, last_frame_id)

        if window_fnames is None:
            return self.empty_output()

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

        if not all_read:
            return self.empty_output()

        wavpath = join(vidname, "audio.wav")
        wav = audio.load_wav(wavpath, self.hparams.sample_rate)
        orig_mel = audio.melspectrogram(wav, self.hparams).T

        if self.wrong_only:
            mel = self.crop_audio_window(orig_mel.copy(), img_name)
            # if (mel.shape[0] != self.syncnet_mel_step_size):
            #     continue
        else:
            mel = orig_mel

        # H x W x 3 * T
        x = np.concatenate(window, axis=2) / 255.0
        x = x.transpose(2, 0, 1)
        x = x[:, x.shape[1] // 2 :]

        # form a bacth
        x = x[: (x.shape[0] // 3 // self.syncnet_T * self.syncnet_T) * 3, :, :]
        x = x.reshape(-1, self.syncnet_T, 3, x.shape[-2], x.shape[-1])
        # reshape to (B, C, T, H, W)
        x = x.transpose(0, 2, 1, 3, 4)

        mel = mel[
            : (mel.shape[0] // self.syncnet_mel_step_size) * self.syncnet_mel_step_size,
            :,
        ]
        mel = mel.reshape(-1, self.syncnet_mel_step_size, mel.shape[-1])
        mel = mel[: x.shape[0], :, :]
        mel = mel.transpose(0, 2, 1)

        x = torch.FloatTensor(x)
        mel = torch.FloatTensor(mel).unsqueeze(1)  # should be B, 1, 80, T

        if x.shape[0] == 0:
            return self.empty_output()

        if x.shape[0] > mel.shape[0]:
            x = x[: mel.shape[0]]

        if self.save_for_debug:
            from Wav2LipMarlin.datasets.video import save_video_with_audio

            video = x.transpose(2, 1)
            video = video.reshape(-1, video.shape[2], video.shape[3], video.shape[4])
            video = (video.permute(0, 2, 3, 1) * 255).int()
            audio_clip = torch.tensor(wav[: int(video.shape[0] / 25 * 16000)])
            save_video_with_audio(video, audio_clip, audio_sample_rate=16000)

        return x, mel, y
