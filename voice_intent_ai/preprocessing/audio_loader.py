import os
import torch
import torchaudio
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

from voice_intent_ai.config import AUDIO_DIR, SAMPLE_RATE, DURATION

class AudioLoader:
    """
    A class for loading and preprocessing audio files.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, target_duration: float = DURATION):
        """
        Initialize the AudioLoader.

        Args:
            sample_rate: Target sample rate for audio files
            target_duration: Target duration in seconds (longer files will be trimmed, shorter ones will be padded)
        """
        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.target_length = int(sample_rate * target_duration)

    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load an audio file and convert it to the target sample rate.

        Args:
            audio_path: Path to the audio file

        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        # Handle relative paths
        if not Path(audio_path).is_absolute():
            audio_path = str(AUDIO_DIR / audio_path)

        try:
            # Try loading with torchaudio first
            waveform, sample_rate = torchaudio.load(audio_path)

            # Convert to mono if needed
            if waveform.dim() > 1 and waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample if needed
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.sample_rate
                )
                waveform = resampler(waveform)
                sample_rate = self.sample_rate

            return waveform, sample_rate

        except Exception as e:
            # Fallback to librosa if torchaudio fails
            try:
                waveform, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                waveform = torch.from_numpy(waveform).unsqueeze(0)  # Add channel dimension
                return waveform, self.sample_rate

            except Exception as e:
                print(f"Error loading audio file {audio_path}: {str(e)}")
                # Return silent audio with correct shape
                return torch.zeros((1, self.target_length)), self.sample_rate

    def process_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load and process an audio file to a fixed length.

        Args:
            audio_path: Path to the audio file

        Returns:
            Processed audio tensor with shape (1, target_length)
        """
        waveform, _ = self.load_audio(audio_path)

        # Ensure correct shape: (channels, samples)
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension

        # Trim or pad the audio to the target length
        if waveform.size(1) > self.target_length:
            # Random crop for training, center crop for validation/test
            start = torch.randint(0, waveform.size(1) - self.target_length + 1, (1,)).item()
            waveform = waveform[:, start:start + self.target_length]
        elif waveform.size(1) < self.target_length:
            # Pad with zeros
            pad_length = self.target_length - waveform.size(1)
            padding = (0, pad_length)  # (left, right) padding
            waveform = torch.nn.functional.pad(waveform, padding)

        # Normalize to [-1, 1]
        if waveform.max() > 0:
            waveform = waveform / (waveform.abs().max() + 1e-8)

        return waveform.squeeze(0)  # Remove channel dimension for most models
