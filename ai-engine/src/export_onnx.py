"""
FinGuard - ONNX Export
Converts both trained models (sklearn Isolation Forest + PyTorch Autoencoder)
into the ONNX format so the backend can run fast, lightweight inference
using only onnxruntime - without needing PyTorch or scikit-learn installed.
"""

import joblib
import torch
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

# Must match the training scripts exactly - same feature order matters!
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
INPUT_DIM = len(FEATURE_COLS)


def export_isolation_forest():
    print("Exporting Isolation Forest to ONNX...")
    model = joblib.load("models/isolation_forest.pkl")

    # Tell ONNX what shape of input to expect:
    # [None, 8] means "any number of rows, 8 features each"
    initial_type = [("input", FloatTensorType([None, INPUT_DIM]))]
    # target_opset pins the ONNX operator versions so they stay compatible
    # with the onnxruntime version we're using for verification/inference.
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset={"": 13, "ai.onnx.ml": 3}
    )

    with open("models/isolation_forest.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    print("Saved models/isolation_forest.onnx")


def export_autoencoder():
    print("Exporting Autoencoder to ONNX...")

    # Re-declare the same architecture used in training so we can load the weights
    from torch import nn

    class Autoencoder(nn.Module):
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
            return self.decoder(self.encoder(x))

    model = Autoencoder(INPUT_DIM)
    model.load_state_dict(torch.load("models/autoencoder.pt"))
    model.eval()

    # ONNX export needs a sample/dummy input to trace the computation graph
    dummy_input = torch.randn(1, INPUT_DIM)

    torch.onnx.export(
        model,
        dummy_input,
        "models/autoencoder.onnx",
        input_names=["input"],
        output_names=["reconstructed"],
        dynamic_axes={
            "input": {0: "batch_size"},        # allow variable batch sizes at inference
            "reconstructed": {0: "batch_size"}
        },
        opset_version=18   # native version for this PyTorch install - avoids a noisy downgrade conversion
    )
    print("Saved models/autoencoder.onnx")


def verify_onnx_models():
    """
    Sanity check: run both ONNX models on a dummy input and confirm
    they produce output without errors - catches export mistakes early.
    """
    print("\nVerifying ONNX models load and run correctly...")
    dummy_input = np.random.randn(3, INPUT_DIM).astype(np.float32)

    if_session = ort.InferenceSession("models/isolation_forest.onnx")
    if_output = if_session.run(None, {"input": dummy_input})
    print(f"Isolation Forest ONNX output shapes: {[o.shape for o in if_output]}")

    ae_session = ort.InferenceSession("models/autoencoder.onnx")
    ae_output = ae_session.run(None, {"input": dummy_input})
    print(f"Autoencoder ONNX output shape: {ae_output[0].shape}")

    print("\nBoth ONNX models verified working!")


if __name__ == "__main__":
    export_isolation_forest()
    export_autoencoder()
    verify_onnx_models()