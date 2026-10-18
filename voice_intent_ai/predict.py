import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Project imports
from models.audio_model import AudioModel
from preprocessing.audio_loader import AudioLoader
from preprocessing.feature_extractor import AudioFeatureExtractor

class AudioIntentPredictor:
    def __init__(self, model_path, sample_rate=16000):
        """Initialize the predictor with a trained model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate

        # Load model
        self.model = AudioModel(num_classes=10).to(self.device)  # Update num_classes as needed
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Initialize audio processors
        self.audio_loader = AudioLoader(sample_rate=sample_rate)
        self.feature_extractor = AudioFeatureExtractor(sample_rate=sample_rate)

        # Example class names (update with your actual classes)
        self.classes = [f"class_{i}" for i in range(10)]

    def predict_file(self, audio_path):
        """Make a prediction for a single audio file."""
        try:
            # Load and preprocess audio
            waveform = self.audio_loader.load_audio(audio_path)

            # Extract features
            features = self.feature_extractor.extract_features(waveform)

            # Convert to tensor and add batch dimension
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred_class = torch.max(probs, 1)

            return {
                'class': self.classes[pred_class.item()],
                'confidence': confidence.item(),
                'probabilities': {c: p.item() for c, p in zip(self.classes, probs.squeeze())}
            }

        except Exception as e:
            return {'error': str(e)}

    def predict_directory(self, audio_dir, extensions=('.wav', '.mp3', '.flac')):
        """Make predictions for all audio files in a directory."""
        audio_dir = Path(audio_dir)
        results = {}

        # Find all audio files
        audio_files = []
        for ext in extensions:
            audio_files.extend(audio_dir.glob(f'*{ext}'))

        # Make predictions
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            results[str(audio_file)] = self.predict_file(audio_file)

        return results

if __name__ == "__main__":
    # Example usage
    model_path = "outputs/models/audio_model.pth"
    audio_path = "path/to/your/audio/file.wav"  # Replace with your audio file or directory

    # Initialize predictor
    predictor = AudioIntentPredictor(model_path)

    # Make prediction
    if Path(audio_path).is_file():
        result = predictor.predict_file(audio_path)
        print(f"\nPrediction for {audio_path}:")
        print(f"Class: {result['class']}")
        print(f"Confidence: {result['confidence']:.4f}")
    elif Path(audio_path).is_dir():
        results = predictor.predict_directory(audio_path)
        print("\nPredictions:")
        for file, pred in results.items():
            print(f"{file}: {pred['class']} (Confidence: {pred['confidence']:.4f})")
    else:
        print(f"Error: {audio_path} is not a valid file or directory")
