import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any

from .audio_loader import AudioLoader


class CommonVoiceDataset(Dataset):
    """
    A PyTorch Dataset class for loading and processing CommonVoice audio data.
    """
    def __init__(self, 
                 manifest_path: str, 
                 audio_dir: str, 
                 sample_rate: int = 16000,
                 max_duration: float = 10.0,
                 min_duration: float = 1.0,
                 transform=None,
                 audio_loader=None):
        """
        Initialize the dataset.
        
        Args:
            manifest_path (str): Path to the manifest file (TSV format)
            audio_dir (str): Directory containing audio files
            sample_rate (int): Target sample rate for audio
            max_duration (float): Maximum duration of audio in seconds
            min_duration (float): Minimum duration of audio in seconds
            transform: Optional transform to be applied to the audio
            audio_loader: Optional AudioLoader instance for loading and processing audio
        """
        self.audio_dir = audio_dir
        self.audio_loader = audio_loader or AudioLoader(sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.transform = transform
        
        # Load and filter the manifest
        self.df = self._load_and_filter_manifest(manifest_path)
        
    def _load_and_filter_manifest(self, manifest_path: str) -> pd.DataFrame:
        """
        Load the manifest file and filter based on duration.
        
        Args:
            manifest_path (str): Path to the manifest file
            
        Returns:
            pd.DataFrame: Filtered DataFrame containing valid samples
        """
        # Read the manifest file
        df = pd.read_csv(manifest_path, sep='\t')
        
        # Filter based on duration if duration column exists
        if 'duration' in df.columns:
            df = df[(df['duration'] >= self.min_duration) & 
                   (df['duration'] <= self.max_duration)]
            
        # Reset index after filtering
        return df.reset_index(drop=True)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (audio_tensor, label) where audio_tensor is a torch.Tensor
        """
        # Get the row from the dataframe
        row = self.df.iloc[idx]
        
        # Get the audio file path
        audio_path = os.path.join(self.audio_dir, row['path'])
        
        # Load the audio file
        # Note: The actual loading will be done by the AudioLoader
        # We'll just return the path and let the collate function handle loading
        
        # Get the label (assuming 'intent' column exists in the manifest)
        label = row.get('intent', 0)  # Default to 0 if 'intent' column doesn't exist
        
        return {
            'audio_path': audio_path,
            'label': label,
            'sample_rate': self.sample_rate
        }

def collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for the DataLoader.
    
    Args:
        batch (list): List of dictionaries containing 'audio_path' and 'label'
        
    Returns:
        tuple: (audio_paths, labels) where:
            - audio_paths is a list of audio file paths
            - labels is a tensor of labels
    """
    # Extract audio paths and labels from the batch
    audio_paths = [item['audio_path'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    
    return audio_paths, labels
