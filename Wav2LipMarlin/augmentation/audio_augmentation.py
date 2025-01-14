import torch
import random
import torchaudio

# Augmentations
def add_random_noise(audio_batch, noise_level=0.05, sample_rate = 16000):
    """
    Add random Gaussian noise to the audio.

    Args:
        audio_batch (torch.Tensor): Tensor of shape (B, T).
        noise_level (float): Standard deviation of the Gaussian noise.

    Returns:
        torch.Tensor: Audio batch with added noise.
    """
    noise = torch.randn_like(audio_batch) * noise_level
    return audio_batch + noise

def scale_amplitude(audio_batch, scale_range=(0.8, 1.2), sample_rate = 16000):
    """
    Scale the amplitude of the audio by a random factor.

    Args:
        audio_batch (torch.Tensor): Tensor of shape (B, T).
        scale_range (tuple): Range of scaling factors (min, max).

    Returns:
        torch.Tensor: Scaled audio batch.
    """
    scale = random.uniform(*scale_range)
    return audio_batch * scale

def pitch_shift(audio_batch, n_steps_range=(-2, 2), sample_rate = 16000):
    """
    Shift the pitch of the audio.

    Args:
        audio_batch (torch.Tensor): Tensor of shape (B, T).
        sample_rate (int): Sampling rate of the audio.
        n_steps_range (tuple): Range of pitch shift steps (min, max).

    Returns:
        torch.Tensor: Pitch-shifted audio batch.
    """
    n_steps = random.uniform(*n_steps_range)
    pitch_shifter = torchaudio.transforms.PitchShift(sample_rate, n_steps=n_steps)
    return torch.stack([pitch_shifter(audio.unsqueeze(0)).squeeze(0) for audio in audio_batch])

class AudioAugmentationPipeline:
    def __init__(self, sample_rate=16000, augmentations=[
        (0.25, add_random_noise),  # 50% probability
        (0.25, scale_amplitude),  # 50% probability
  #      (0.25, pitch_shift),  # 50% probability
    ]):
        """
        Initialize the audio augmentation pipeline.

        Args:
            sample_rate (int): Sampling rate of the audio.
            augmentations (list of tuples): Each tuple is (probability, augmentation_fn).
        """
        self.sample_rate = sample_rate
        self.augmentations = augmentations if augmentations is not None else []

    def add_augmentation(self, p, augmentation_fn):
        """
        Add an augmentation function to the pipeline.

        Args:
            p (float): Probability of applying this augmentation.
            augmentation_fn (callable): A function that takes a tensor and returns an augmented tensor.
        """
        self.augmentations.append((p, augmentation_fn))

    def __call__(self, audio_batch):
        """
        Apply the augmentation pipeline to a batch of audio.

        Args:
            audio_batch (torch.Tensor): Tensor of shape (B, T).

        Returns:
            torch.Tensor: Augmented audio batch.
        """
        B = audio_batch.shape[0]
        for p, aug in self.augmentations:
            # Select a random subset of batch elements
            batch_indices = torch.randperm(B)[:random.randint(1, int(p * B))]  # Random number of elements
            for idx in batch_indices:
                audio_batch[idx] = aug(audio_batch[idx], sample_rate=self.sample_rate)
        return audio_batch

# Usage example
if __name__ == "__main__":
    # Create a random audio batch of shape (2, 16000) simulating 1 second of audio at 16 kHz
    sample_rate = 16000
    audio_batch = torch.randn(2, 16000)

    # Initialize the pipeline with some augmentations
    pipeline = AudioAugmentationPipeline(sample_rate=sample_rate, augmentations=[
        (0.5, lambda x: add_random_noise(x, noise_level=0.05)),  # 50% probability
        (0.5, lambda x: scale_amplitude(x, scale_range=(0.8, 1.2))),  # 50% probability
        (0.5, pitch_shift),  # 50% probability
    ])

    # Apply the pipeline
    augmented_audio = pipeline(audio_batch)

    print("Original shape:", audio_batch.shape)
    print("Augmented shape:", augmented_audio.shape)
