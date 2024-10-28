
import os
from os.path import join
from glob import glob
import tqdm

def get_image_list(data_root, split, manifest_root):
    filelist = []

    with open(os.path.join(manifest_root, '{}.txt'.format(split))) as f:
        for line in f:
            line = line.strip()
            if ' ' in line: line = line.split()[0]
            filelist.append(os.path.join(data_root, line))

    return filelist

def get_sorted_names(vidname, img_names):
    # get sorted names of files
    sorted_available_ids = sorted([int(_.split('.')[0].split('/')[-1]) for _ in img_names])
    sorted_files = [str(_)+'.jpg' for _ in sorted_available_ids]
    img_names = [join(vidname, _) for _ in sorted_files]
    return img_names, sorted_available_ids

def main(data_root = '/data/LRS2/main/', syncnet_T = 16, manifest_root='./filelists', split = 'train'):
    
    all_videos = get_image_list(data_root, split, manifest_root=manifest_root)
    
    short_video_idx = []
    extremelly_short_idx = []
    long_videos_idx = []
    medium_videos_idx = []
    
    all_videos_mapping = {}
    for idx in tqdm.tqdm(range(len(all_videos))):
        vidname = all_videos[idx]
        img_names = list(glob(join(vidname, '*.jpg')))
        
        video_len_type = None
        if len(img_names) < syncnet_T:
            extremelly_short_idx.append(idx)
            video_len_type = 'ext_short'
            # exclude from dataset
            continue
        if len(img_names) <= 2 * syncnet_T:
            short_video_idx.append(idx)
            video_len_type = 'short' # if short replace audio
        elif len(img_names) >= 3* syncnet_T:
            long_videos_idx.append(idx)
            video_len_type = 'long' # default logic
        else:
            medium_videos_idx.append(idx)
            video_len_type = 'medium' # fixed offset
        
        sorted_img_names, sorted_available_ids = get_sorted_names(vidname, img_names)
        all_videos_mapping[idx] = (video_len_type, sorted_img_names, sorted_available_ids)
        continue
    return

if __name__ == "__main__":
    main()