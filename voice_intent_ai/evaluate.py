import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Project imports
from models.audio_model import AudioModel
from preprocessing.audio_loader import AudioLoader
from preprocessing.data_processor import CommonVoiceDataset
from utils.visualization import plot_confusion_matrix

def evaluate(model_path, test_manifest, audio_dir):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize components
    audio_loader = AudioLoader(sample_rate=16000)
    test_dataset = CommonVoiceDataset(
        manifest_path=test_manifest,
        audio_dir=audio_dir,
        audio_loader=audio_loader
    )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load model
    model = AudioModel(num_classes=len(test_dataset.label_encoder.classes_)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Evaluation
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for audio, labels in tqdm(test_loader, desc="Evaluating"):
            audio = audio.to(device)
            outputs = model(audio)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = (all_preds == all_labels).mean()
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=test_dataset.label_encoder.classes_
    ))

    # Confusion matrix
    os.makedirs("outputs/figures", exist_ok=True)
    plot_confusion_matrix(
        all_labels,
        all_preds,
        class_names=test_dataset.label_encoder.classes_,
        save_path="outputs/figures/confusion_matrix.png"
    )
    print("Confusion matrix saved to outputs/figures/confusion_matrix.png")

if __name__ == "__main__":
    model_path = "outputs/models/audio_model.pth"
    test_manifest = "data/raw/test.tsv"
    audio_dir = "data/raw/clips"

    evaluate(model_path, test_manifest, audio_dir)
