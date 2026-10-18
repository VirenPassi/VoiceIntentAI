import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
from voice_intent_ai.preprocessing.data_processor import collate_fn, CommonVoiceDataset

# Add the project root to Python path to ensure imports work in both direct and module execution
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Project imports
from voice_intent_ai.models.audio_model import AudioModel
from voice_intent_ai.preprocessing.data_processor import CommonVoiceDataset

# Configuration
config = {
    "data": {
        "train_manifest": "voice_intent_ai/data/raw/train.tsv",
        "test_manifest": "voice_intent_ai/data/raw/test.tsv",
        "audio_dir": "voice_intent_ai/data/raw/clips",
        "max_duration": 10.0,
        "min_duration": 1.0
    },
    "audio": {"sample_rate": 16000},
    "training": {
        "batch_size": 32,
        "num_epochs": 10,
        "learning_rate": 1e-4
    },
    "output": {
        "model_dir": "outputs/models",
        "results_dir": "outputs/results"
    }
}

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize dataset
train_dataset = CommonVoiceDataset(
    manifest_path=config["data"]["train_manifest"],
    audio_dir=config["data"]["audio_dir"],
    sample_rate=config["audio"]["sample_rate"],
    max_duration=config["data"]["max_duration"],
    min_duration=config["data"]["min_duration"]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    collate_fn=collate_fn
)

# Determine number of classes (assuming 'intent' column exists in the manifest)
num_classes = len(train_dataset.df['intent'].unique()) if 'intent' in train_dataset.df.columns else 10
model = AudioModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

# Training loop
for epoch in range(config["training"]["num_epochs"]):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch_idx, (audio_paths, labels) in enumerate(pbar):
        # Calculate target length based on max_duration and sample_rate
        target_length = int(config["audio"]["sample_rate"] * config["data"]["max_duration"])
        
        # Load and process audio files
        audio_batch = []
        for audio_path in audio_paths:
            try:
                # Load audio file
                audio, orig_sample_rate = torchaudio.load(audio_path)
                
                # Resample if needed
                if orig_sample_rate != config["audio"]["sample_rate"]:
                    audio = torchaudio.functional.resample(
                        audio, 
                        orig_freq=orig_sample_rate,
                        new_freq=config["audio"]["sample_rate"]
                    )
                
                # Pad or truncate to target length
                if audio.size(1) < target_length:
                    # Pad with zeros if audio is shorter
                    pad_amount = target_length - audio.size(1)
                    audio = torch.nn.functional.pad(audio, (0, pad_amount))
                else:
                    # Truncate if audio is longer
                    audio = audio[:, :target_length]
                
                audio_batch.append(audio)
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                # Add a zero tensor of the correct shape if loading fails
                audio_batch.append(torch.zeros(1, target_length))
        
        # Stack audio tensors
        audio_tensor = torch.cat(audio_batch, dim=0)  # [batch_size, time]
        audio_tensor = audio_tensor.unsqueeze(1)  # Add channel dimension [batch_size, 1, time]
        
        # Move to device
        audio_tensor = audio_tensor.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(audio_tensor)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })

# Create output directories
os.makedirs(config["output"]["model_dir"], exist_ok=True)
os.makedirs(config["output"]["results_dir"], exist_ok=True)

# Save model
model_path = os.path.join(config["output"]["model_dir"], "audio_model.pth")
torch.save(model.state_dict(), model_path)
print(f"\nTraining completed! Model saved to {model_path}")

# Evaluate on test set
print("\nEvaluating on test set...")
test_dataset = CommonVoiceDataset(
    manifest_path=config["data"]["test_manifest"],
    audio_dir=config["data"]["audio_dir"],
    sample_rate=config["audio"]["sample_rate"],
    max_duration=config["data"]["max_duration"],
    min_duration=config["data"]["min_duration"]
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=False,
    collate_fn=collate_fn
)

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for audio_paths, labels in tqdm(dataloader, desc="Testing"):
            # Load and process audio files
            audio_batch = []
            for audio_path in audio_paths:
                try:
                    audio, orig_sample_rate = torchaudio.load(audio_path)
                    if orig_sample_rate != config["audio"]["sample_rate"]:
                        audio = torchaudio.functional.resample(
                            audio, 
                            orig_freq=orig_sample_rate,
                            new_freq=config["audio"]["sample_rate"]
                        )
                    target_length = int(config["audio"]["sample_rate"] * config["data"]["max_duration"])
                    if audio.size(1) < target_length:
                        pad_amount = target_length - audio.size(1)
                        audio = torch.nn.functional.pad(audio, (0, pad_amount))
                    else:
                        audio = audio[:, :target_length]
                    audio_batch.append(audio)
                except Exception as e:
                    print(f"Error loading {audio_path}: {e}")
                    audio_batch.append(torch.zeros(1, target_length))
            
            audio_tensor = torch.cat(audio_batch, dim=0).unsqueeze(1).to(device)
            labels = labels.to(device)
            
            outputs = model(audio_tensor)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy

# Run evaluation
test_accuracy = evaluate(model, test_loader, device)
print(f"\nTest Accuracy: {test_accuracy:.2f}%")

# Save results
results_path = os.path.join(config["output"]["results_dir"], "test_results.txt")
with open(results_path, 'w') as f:
    f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
    f.write(f"Model saved to: {model_path}\n")
    f.write(f"Number of test samples: {len(test_dataset)}\n")

print(f"\nResults saved to {results_path}")
