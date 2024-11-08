import pytorch_lightning as pl
from torch.utils.data import DataLoader
from SyncNetPLModel import LipSyncExpertModel
from Wav2LipMarlin.datasets.dataset import FullVideoSyncDataset

from Wav2LipMarlin.hparams import hparams

from pytorch_lightning.loggers import TensorBoardLogger



# Create a TensorBoard logger
logger = TensorBoardLogger(
    save_dir='./logs/prediction_logs',   # Directory where logs will be saved
    name='evaluation',            # Name of the experiment (log folder name)
)


def plot_binned_histogram(scores, experiment_name = 'unsynced'):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    # Convert scores to numpy for histogram plotting
    scores = torch.stack(scores, 0)
    scores = scores.cpu().numpy()
    # Define bins for discrete intervals (e.g., 10% intervals)
    bins = np.array([0, 0.05, 0.1, 0.25, 0.45, 0.5, 0.55, 0.75, 0.9, 0.95, 1])
    bins = np.arange(0, 1, 0.05)
    #bins = None
    plt.hist(scores, bins=bins, edgecolor='black', alpha=0.7)
    plt.ylim(0, scores.shape[0]+5)  # Set y-axis range if provided
    if bins is not None:
        plt.xticks(bins, [f"{b}%" for b in bins], rotation=45)  # Label x-axis in percentages
    plt.title(f"Sync-Scores(1-cos_distance) of {experiment_name} audio-videos")
    plt.xlabel("Percentiles(distance)")
    plt.ylabel(f"Number of samples (total: {scores.shape[0]})")
    # Save to file
    plt.savefig(experiment_name+'.png', format='png')
    # Close the plot to free memory
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score

def plot_binned_histogram_with_rocauc(scores, true_labels, experiment_name='unsynced'):
    """
    Plot binned histogram of the scores and compute ROC-AUC curve.
    
    Parameters:
        scores (torch.Tensor): Tensor of predicted scores.
        true_labels (torch.Tensor): Tensor of true labels (binary).
        experiment_name (str): The name for saving the plot.
    """
    # Convert scores to numpy for histogram plotting
    scores = torch.stack(scores, 0)
    scores = scores.cpu().numpy()
    true_labels = true_labels.cpu().numpy()  # Convert true labels to numpy  
    # Calculate ROC-AUC
    auc_score = roc_auc_score(true_labels, scores)
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random classifier')
    plt.title(f'ROC Curve - {experiment_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f'{experiment_name}_roc_curve.png', format='png')
    plt.close()
    # Optionally, print the AUC score
    print(f"ROC-AUC Score for {experiment_name}: {auc_score:.4f}")


def main():
    # Argument Parsing
    import argparse
    parser = argparse.ArgumentParser(description='Run evaluation of SyncNet on dataset')
    parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)
    parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', default=None, type=str)
    parser.add_argument('--accelerator', help='Device type', default="gpu", type=str)
    parser.add_argument('--devices', help='Device ids', default=1, type=int)
    parser.add_argument('--test_original', help='Perform testing of an original SyncNet from Wav2Lip', default=False, type=bool)

    args = parser.parse_args()
    
    if not args.test_original:
        # Params for Syncnet-Marlin 
        syncnet_T = 16
        syncnet_mel_step_size = int(16 * 3.18)
        syncnet_type='marlin'
    else:
        # Params for Syncnet-orginal
        syncnet_T = 5
        syncnet_mel_step_size = 16
        syncnet_type = 'original'
        hparams.img_size = 96
    
    checkpoint_path = args.checkpoint_path

    # Initialize your model
    model = LipSyncExpertModel(hparams, syncnet_T, syncnet_mel_step_size, checkpoint_path, syncnet_type = syncnet_type)
    accelerator = args.accelerator
    if accelerator == 'gpu':
        import torch
        strategy = "ddp_find_unused_parameters_true"
        devices = args.devices
        torch.compile(model.model)
    else:
        strategy = None
        devices = None
    
    # Create a Trainer object
    trainer = pl.Trainer(
        accelerator = accelerator,
        devices = devices,
        strategy = strategy,
        precision = '16-mixed',
        logger = logger,
        limit_predict_batches=0.5  # Run predictions on only 10% of the batches
    )
    
    # Dataset and Dataloader setup and run prediction
    
    # Running synced test
    predict_dataset_correct = FullVideoSyncDataset(args.data_root, 'val', hparams, syncnet_T, syncnet_mel_step_size, 
        wrong_only = False)
    # Wrap the data in a DataLoader (PyTorch Lightning requires DataLoader)
    predict_data_loader = DataLoader(predict_dataset_correct, batch_size=1, shuffle=False,
        num_workers=hparams.num_workers)
    # Run predictions using Trainer.predict
    predictions = trainer.predict(model, predict_data_loader)
    predictions = [_ for _ in predictions if _ is not None]
    predictions_pos = predictions
    predictions_pos_labels = torch.ones(len(predictions))
    plot_binned_histogram(predictions, 'synced')
    
    # Running unsynced test with a range of offsets
    offsets = [1, 3, 5, 7, 15, 25]
    
    for current_offset in offsets:
        predict_dataset_wrong = FullVideoSyncDataset(args.data_root, 'val', hparams, syncnet_T, syncnet_mel_step_size, 
            wrong_only = True, offset=current_offset)
        # Wrap the data in a DataLoader (PyTorch Lightning requires DataLoader)
        predict_data_loader = DataLoader(predict_dataset_wrong, batch_size=1, shuffle=False,
            num_workers=hparams.num_workers)
        # Run predictions using Trainer.predict
        predictions = trainer.predict(model, predict_data_loader)
        predictions = [_ for _ in predictions if _ is not None]
        predictions_negs = predictions
        predictions_negs_labels = torch.zeros(len(predictions))
        plot_binned_histogram(predictions, 'unsynced')
        
        plot_binned_histogram_with_rocauc(
            predictions_pos+predictions_negs, 
            torch.cat([predictions_pos_labels, predictions_negs_labels]), 
            f'mismatch_offset {current_offset}'
        )

if __name__ == "__main__":
    main()
