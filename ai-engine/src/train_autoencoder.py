"""
FinGuard - Autoencoder Training (PyTorch)
Trains a neural network to reconstruct NORMAL transaction patterns only.
When it later sees a fraud transaction, reconstruction error spikes -
that error becomes our second anomaly score, complementing Isolation Forest.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib

FEATURE_COLS = [
    "amount",
    "txn_count_1h",
    "unique_receivers_1h",
    "total_sent_1h",
    "avg_amount_sender",
    "amount_deviation",
    "near_threshold_flag",
    "in_out_ratio",
]

INPUT_DIM = len(FEATURE_COLS)   # 8
EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 0.001


class Autoencoder(nn.Module):
    """
    Symmetric encoder-decoder network.
    8 -> 6 -> 4 -> 2 (bottleneck) -> 4 -> 6 -> 8

    ReLU activations let the network learn non-linear relationships
    between features (e.g. "high txn_count AND near_threshold together
    is worse than either alone") - something a linear model would miss.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 6), nn.ReLU(),
            nn.Linear(6, 4), nn.ReLU(),
            nn.Linear(4, 2), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 4), nn.ReLU(),
            nn.Linear(4, 6), nn.ReLU(),
            nn.Linear(6, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def load_features(path="data/features.csv"):
    return pd.read_csv(path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading feature data...")
    df = load_features()
    X = df[FEATURE_COLS].fillna(0).values
    y_true = df["is_fraud"].values

    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KEY IDEA: train only on NORMAL transactions.
    # The network never sees fraud during training - it only learns
    # what "normal" looks like, so fraud becomes hard to reconstruct.
    X_normal = X_scaled[y_true == 0]
    print(f"Training on {len(X_normal)} normal transactions only")

    X_train_tensor = torch.tensor(X_normal, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, X_train_tensor),
        batch_size=BATCH_SIZE, shuffle=True
    )

    model = Autoencoder(INPUT_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Training autoencoder...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} - Reconstruction loss: {avg_loss:.4f}")

    # ---- Evaluate on FULL dataset (normal + fraud) ----
    print("\nComputing reconstruction error on full dataset...")
    model.eval()
    X_full_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        reconstructed_full = model(X_full_tensor).cpu().numpy()

    # Per-row reconstruction error = mean squared error across features.
    # Higher error = the network struggled to reconstruct it = more anomalous.
    reconstruction_error = np.mean((X_scaled - reconstructed_full) ** 2, axis=1)
    df["ae_anomaly_score"] = reconstruction_error

    # Threshold: flag top ~3% highest errors as fraud (matches our known fraud ratio)
    threshold = np.percentile(reconstruction_error, 97)
    predicted_fraud = (reconstruction_error >= threshold).astype(int)
    df["ae_predicted_fraud"] = predicted_fraud

    print("\n--- Evaluation against known fraud labels ---")
    print(classification_report(y_true, predicted_fraud, target_names=["Normal", "Fraud"]))
    print(f"ROC-AUC Score: {roc_auc_score(y_true, reconstruction_error):.4f}")

    # Save model, scaler, and scored data
    torch.save(model.state_dict(), "models/autoencoder.pt")
    joblib.dump(scaler, "models/ae_scaler.pkl")
    df.to_csv("data/features_scored_ae.csv", index=False)

    print("\nSaved model to models/autoencoder.pt")
    print("Saved scaler to models/ae_scaler.pkl")
    print("Saved scored data to data/features_scored_ae.csv")


if __name__ == "__main__":
    main()