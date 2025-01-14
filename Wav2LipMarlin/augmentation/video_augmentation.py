import torch
import random

# Example augmentation: Duplicate a random frame to all T steps.
def expand_random_frame(video_batch):
    """
    Duplicate a randomly selected frame along the temporal dimension.

    Args:
        video_batch (torch.Tensor): Tensor of shape (B, C, T, H, W).

    Returns:
        torch.Tensor: Tensor with a single frame expanded along the T dimension.
    """
    B, C, T, H, W = video_batch.shape
    random_frame_idx = random.randint(0, T - 1)  # Choose a random frame index
    random_frame = video_batch[:, :, random_frame_idx, :, :].unsqueeze(2)  # Shape (B, C, 1, H, W)
    return random_frame.expand(B, C, T, H, W)  # Expand to (B, C, T, H, W)

class VideoAugmentationPipeline:
    def __init__(self, augmentations=[
        (0.25, expand_random_frame),  # 50% probability
    ]):
        """
        Initialize the augmentation pipeline.

        Args:
            augmentations (list of tuples): Each tuple is (probability, augmentation_fn).
        """
        self.augmentations = augmentations if augmentations is not None else []

    def add_augmentation(self, p, augmentation_fn):
        """
        Add an augmentation function to the pipeline.

        Args:
            p (float): Probability of applying this augmentation.
            augmentation_fn (callable): A function that takes a tensor and returns an augmented tensor.
        """
        self.augmentations.append((p, augmentation_fn))

    def __call__(self, video_batch):
        """
        Apply the augmentation pipeline to a batch of videos.

        Args:
            video_batch (torch.Tensor): Tensor of shape (B, C, T, H, W).

        Returns:
            torch.Tensor: Augmented video batch.
        """
        B, C, T, H, W = video_batch.shape

        for p, aug in self.augmentations:
            # Select a random subset of batch elements
            batch_indices = torch.randperm(B)[:random.randint(1, int(p * B))]  # Random number of elements
            for idx in batch_indices:
                video_batch[idx] = aug(video_batch[idx].unsqueeze(0)).squeeze(0)
        return video_batch

# Usage example
if __name__ == "__main__":
    # Create a random video batch of shape (2, 3, 5, 64, 64)
    video_batch = torch.randn(2, 3, 5, 64, 64)

    # Initialize the pipeline with the expand_random_frame augmentation
    pipeline = VideoAugmentationPipeline(augmentations=[
        (0.5, expand_random_frame),  # 50% probability
    ])

    # Apply the pipeline
    augmented_videos = pipeline(video_batch)

    print("Original shape:", video_batch.shape)
    print("Augmented shape:", augmented_videos.shape)
