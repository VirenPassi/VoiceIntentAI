import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import librosa.display
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def plot_waveform(waveform: np.ndarray,
                 sample_rate: int,
                 title: str = 'Waveform',
                 save_path: Optional[Union[str, Path]] = None) -> None:
    """Plot the waveform of an audio signal.

    Args:
        waveform: Audio signal as a numpy array
        sample_rate: Sample rate of the audio
        title: Title for the plot
        save_path: If provided, save the plot to this path
    """
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(waveform, sr=sample_rate)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_spectrogram(spectrogram: np.ndarray,
                   sample_rate: int,
                   hop_length: int,
                   title: str = 'Spectrogram',
                   y_axis: str = 'linear',
                   save_path: Optional[Union[str, Path]] = None) -> None:
    """Plot a spectrogram.

    Args:
        spectrogram: Spectrogram as a numpy array (freq_bins, time_frames)
        sample_rate: Sample rate of the audio
        hop_length: Hop length used in STFT
        title: Title for the plot
        y_axis: Type of y-axis scaling ('linear', 'log', 'mel', etc.)
        save_path: If provided, save the plot to this path
    """
    plt.figure(figsize=(14, 8))
    librosa.display.specshow(
        spectrogram,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis='time',
        y_axis=y_axis,
        cmap='viridis'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_mfcc(mfcc: np.ndarray,
             sample_rate: int,
             hop_length: int,
             title: str = 'MFCC',
             save_path: Optional[Union[str, Path]] = None) -> None:
    """Plot MFCC features.

    Args:
        mfcc: MFCC features as a numpy array (n_mfcc, time_frames)
        sample_rate: Sample rate of the audio
        hop_length: Hop length used in MFCC computation
        title: Title for the plot
        save_path: If provided, save the plot to this path
    """
    plt.figure(figsize=(14, 6))
    librosa.display.specshow(
        mfcc,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis='time',
        cmap='coolwarm'
    )
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_attention_weights(attention_weights: np.ndarray,
                         title: str = 'Attention Weights',
                         xlabel: str = 'Time Steps',
                         ylabel: str = 'Attention Weights',
                         save_path: Optional[Union[str, Path]] = None) -> None:
    """Plot attention weights.

    Args:
        attention_weights: Attention weights as a numpy array (n_heads, seq_len) or (seq_len,)
        title: Title for the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        save_path: If provided, save the plot to this path
    """
    plt.figure(figsize=(14, 6))

    if len(attention_weights.shape) == 1:
        # Single attention head
        plt.plot(attention_weights, 'b-', linewidth=2)
    else:
        # Multiple attention heads
        for i in range(attention_weights.shape[0]):
            plt.plot(attention_weights[i], '-', alpha=0.7, label=f'Head {i+1}')
        plt.legend()

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_training_history(history: Dict[str, List[float]],
                        metrics: List[str] = ['loss', 'accuracy'],
                        val_metrics: List[str] = ['val_loss', 'val_accuracy'],
                        save_path: Optional[Union[str, Path]] = None) -> None:
    """Plot training history.

    Args:
        history: Dictionary containing training history
        metrics: List of metrics to plot from training
        val_metrics: List of validation metrics to plot
        save_path: If provided, save the plot to this path
    """
    num_plots = len(metrics)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))

    if num_plots == 1:
        axes = [axes]

    for i, (metric, val_metric) in enumerate(zip(metrics, val_metrics)):
        if metric in history and val_metric in history:
            epochs = range(1, len(history[metric]) + 1)

            axes[i].plot(epochs, history[metric], 'b-', label=f'Training {metric}')
            axes[i].plot(epochs, history[val_metric], 'r-', label=f'Validation {val_metric}')

            axes[i].set_title(f'Training and Validation {metric.title()}')
            axes[i].set_xlabel('Epochs')
            axes[i].set_ylabel(metric.title())
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_confusion_matrix(y_true: Union[np.ndarray, List],
                         y_pred: Union[np.ndarray, List],
                         class_names: List[str],
                         normalize: bool = True,
                         title: str = 'Confusion Matrix',
                         save_path: Optional[Union[str, Path]] = None) -> None:
    """Plot a confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize the confusion matrix
        title: Title for the plot
        save_path: If provided, save the plot to this path
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_classification_report(y_true: Union[np.ndarray, List],
                             y_pred: Union[np.ndarray, List],
                             class_names: List[str],
                             save_path: Optional[Union[str, Path]] = None) -> None:
    """Plot a classification report as a heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: If provided, save the plot to this path
    """
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).iloc[:-1, :].T

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        report_df,
        annot=True,
        cmap='viridis',
        fmt='.2f',
        cbar=False,
        linewidths=0.5
    )

    plt.title('Classification Report')
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
