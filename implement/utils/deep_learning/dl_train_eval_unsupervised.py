import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve
)
import matplotlib.pyplot as plt


def compute_reconstruction_errors(model, data_loader, device='cpu'):
    """Compute per-sequence mean MSE reconstruction error."""
    model.eval()
    errors = []
    with torch.no_grad():
        for (batch_x,) in data_loader:
            batch_x = batch_x.to(device)
            recon_x = model(batch_x)
            # MSE per sample across (seq_len, input_dim)
            sample_mse = F.mse_loss(recon_x, batch_x, reduction='none').mean(dim=[1, 2])
            errors.extend(sample_mse.cpu().numpy())
    return np.array(errors)


def train_and_evaluate_dl_unsupervised(
    models,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    epochs=50,
    batch_size=64,
    lr=1e-3,
    device='cpu',
    models_dir=None,
    plots_dir=None,
    patience=7,
    k_threshold=3.0
):
    """
    Trains DL autoencoders exclusively on normal training data (y_train == 0).
    Uses early stopping patience=7 on normal validation data.
    Computes dynamic threshold = mean + k_threshold * std on normal validation data.
    Evaluates on test set (mixed normal DJI and spoofed ESP32).
    """
    # Filter training and validation sets to normal data only (y == 0)
    X_train_normal = X_train[y_train == 0]
    X_val_normal = X_val[y_val == 0]

    train_tensor = torch.tensor(X_train_normal, dtype=torch.float32)
    val_tensor = torch.tensor(X_val_normal, dtype=torch.float32)
    test_tensor = torch.tensor(X_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_tensor), batch_size=batch_size, shuffle=False)

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name} on {len(X_train_normal)} normal sequences...")
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(device)
                optimizer.zero_grad()
                recon_x = model(batch_x)
                loss = F.mse_loss(recon_x, batch_x)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_x)

            train_loss /= len(train_tensor)

            # Validation step
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (batch_x,) in val_loader:
                    batch_x = batch_x.to(device)
                    recon_x = model(batch_x)
                    loss = F.mse_loss(recon_x, batch_x)
                    val_loss += loss.item() * len(batch_x)

            val_loss /= len(val_tensor)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch} (Best Val Loss: {best_val_loss:.6f})")
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model = model.to(device)

        # Compute dynamic threshold on normal validation set
        val_errors = compute_reconstruction_errors(model, val_loader, device=device)
        threshold = val_errors.mean() + k_threshold * val_errors.std()

        # Evaluate on test set
        test_errors = compute_reconstruction_errors(model, test_loader, device=device)
        y_pred = (test_errors > threshold).astype(int)

        prec_vals, rec_vals, _ = precision_recall_curve(y_test, test_errors)
        pr_auc = auc(rec_vals, prec_vals)

        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, test_errors),
            'pr_auc': pr_auc,
            'threshold': float(threshold),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

        if models_dir is not None:
            models_dir = Path(models_dir)
            models_dir.mkdir(parents=True, exist_ok=True)
            model_filename = f"{name.lower().replace(' ', '_').replace('-', '_')}.pth"
            torch.save(model.state_dict(), models_dir / model_filename)

    if plots_dir is not None:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 8))
        for name, model in models.items():
            model = model.to(device)
            test_errors = compute_reconstruction_errors(model, test_loader, device=device)
            fpr, tpr, _ = roc_curve(y_test, test_errors)
            roc_auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_val:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Deep Learning Autoencoders ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(plots_dir / 'autoencoder_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    return results
