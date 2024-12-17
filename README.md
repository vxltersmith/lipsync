<h3 align="center">Wav2Lip + MARLIN</h3>

<p align="center"> CNN for generation of lipsynced videos: training and inference basic implementation
    <br> 
</p>

---

### 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [TODO](#todo)
- [Contributing](#contributing)
- [Authors](#authors)
- [Acknowledgments](#acknowledgements)

---

### 🧐 About <a name="about"></a>

Re-implementation of **Wav2Lip + MARLIN**, mentioned in the paper: [https://arxiv.org/abs/2211.06627](https://arxiv.org/abs/2211.06627).  
The project builds upon the work of:
- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
- [MARLIN](https://github.com/ControlNet/MARLIN) - pre-trained ViT face video encoder

It utilizes **XLSR-53** as the audio encoder, **InceptionResnetV1** as identity and style loss and leverages **Pytorch-Lightning** for structured training and inference.

---

### 🏁 Getting Started <a name="getting_started"></a>

To set up this project, you will need:
1. **NVIDIA NGC PyTorch container** (or PyTorch>=2.0 installed manually).
2. Dependencies installed via `requirements.txt`.
3. OpenGL dependencies for **cv2** (OpenCV, needed to stabilize dependencies).
4. **Real-ESRGAN** requirements must be manually installed from it's subfolder to be able to use it for video super-resolution in inference.
---

### ⚙️ Installation <a name="installation"></a>

Follow these steps to build and prepare the environment:

#### 1. Pull NVIDIA NGC PyTorch Container
Ensure you have Docker installed. Pull the NVIDIA PyTorch container optimized for deep learning:

```bash
docker pull nvcr.io/nvidia/pytorch:<tag>  # Replace <tag> with the required PyTorch version, e.g., 23.06-py3
```

2. Install Requirements

Clone the repository and install all Python dependencies:

```bash
git clone https://github.com/vxltersmith/lipsync.git
cd lipsync
pip install -r requirements.txt
```

3. Install OpenGL Dependencies for OpenCV

Run the provided script to install OpenCV dependencies that rely on OpenGL:
```bash
bash cv2_install_dependency.sh
```
Ensure the system has OpenGL and related libraries required for cv2 (OpenCV).

4. Install Real-ESRGAN Requirements
```bash
cd ./Real-ESRGAN
pip install -r requirements.txt
```

After this step might be neccessary to repeat steps 2 and 3:
```bash
cd lipsync
pip install --force -r requirements.txt
bash cv2_install_dependency.sh
```

Usage <a name="usage"></a>

You can run inference to generate lipsynced videos by calling the following script:
Inference Command:

```bash
python infer_wav2lip_marlin.py \
    --checkpoint_path ./checkpoints/wav2lip_marlin/audio_emb_lrsmain_l1_id_sync_spec_norm_drop/l1_vl_checkpoint_epoch=116_step=335205.ckpt \
    --source_video_path ./toloka_test_v0/monologues_renamed_25fps/alex_danvers_25fps.mp4 \
    --audio_path ./toloka_test_v0/audio_targets/obama_1min.wav \
    --output_dir ./toloka_test_v0/generated_videos/ \
    --start_at 130 \
    --audio_encoder_type wav2vec
```
Parameters:

    --checkpoint_path: Path to the model checkpoint.
    --source_video_path: Path to the input video file.
    --audio_path: Path to the target audio file(audio can be extracted from .mp4).
    --output_dir: Directory to save the generated video.
    --start_at: Time in seconds to start processing the source video.
    --audio_encoder_type: Audio encoder type (e.g., wav2vec).

### 🚀 Training <a name="training"></a>

Training and debugging was made on LRS2 dataset, preparation procedure was taken from original Wav2Lip repo in mostly original implementation

To preprocess LRS2 via script from this repo run:
```bash
python preprocess.py \
    --ngpu 2 \
    --batch_size 32 \
    --data_root /path/to/LRS2/dataset/ \
    --preprocessed_root /path/to/preprocessed_data/ \
    --check_before_skip True
```
Explanation of the arguments:

    --ngpu: Specifies the number of GPUs you want to use for face cropping. In this case, it's set to 2, which means the script will utilize two GPUs.
    --batch_size: Defines the batch size for face detection when running on a single GPU. Here, it’s set to 32.
    --data_root: Path to the root directory where the LRS2 dataset (or any other dataset) is stored. Replace /path/to/LRS2/dataset/ with the actual path to your dataset.
    --preprocessed_root: Directory to store the preprocessed data. Replace /path/to/preprocessed_data/ with the desired path where the processed files will be saved.
    --check_before_skip: If set to True, the script will check if a file has already been processed and skip it, preventing redundant processing.

You can train the model on your dataset using the following command:

Training Command:
```bash
python train_wav2lip_marlin_pl.py \
    --data_root /data/LRS2/main/ \
    --checkpoint_dir full_reference_all_losses_v0 \
    --syncnet_checkpoint_path ./checkpoints/Wav2Lip/lipsync_expert.pth \
    --batch_size 16 \
    --checkpoint_path ./checkpoints/wav2lip_marlin/full_reference_all_losses_v0/sl_checkpoint_epoch=54_step=157575.ckpt \
    --sync_weight 0.03 \
    --is_weight 0.07
```
Parameters:

    --data_root: Path to the dataset root directory.
    --checkpoint_dir: Directory to save training checkpoints. Curretly this also used as experiment name for logging
    --syncnet_checkpoint_path: Path to the pre-trained SyncNet checkpoint for lipsync supervision. Required but only used with default audio encoder
    --batch_size: Batch size for training.
    --sync_weight: Weight for the sync loss.
    --is_weight: Weight for the identity loss.

### TODO <a name = "TODO"></a>

Currently no lypsinc expert was used for training and temporal loss might be a major improvement which would make this work close to original Wav2Lip. Repo must be considered as semi-stable and still need further improvement, but still can train and infere model in both default and Wav2Vec audio encoder modes. Improvements mostly needed to achieve better quality and consistency with original Wav2Lip

### ✍️ Authors <a name = "authors"></a>

- [@vxltersmith](https://github.com/vxltersmith) - Original author

See also the list of [contributors](https://github.com/vxltersmith/The-Documentation-Compendium/contributors) who participated in this project.
 - Curretly there is no additional contributors and project might be opened for that

### 🎉 Acknowledgements <a name = "acknowledgement"></a>

- Great work made it possible:
  - https://github.com/xinntao/Real-ESRGAN
  - https://github.com/ControlNet/MARLIN
  - https://github.com/Rudrabha/Wav2Lip
- Thanks to 
  - https://github.com/xinntao
  - https://github.com/ControlNet
  - https://github.com/Rudrabha
- References
  - https://arxiv.org/abs/2211.06627.
  - https://arxiv.org/abs/2008.10010
