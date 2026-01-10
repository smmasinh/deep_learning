import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


TRAIN_FILE = "train_points.csv"  
TEST_FILE = "test_points.csv"  
OUTPUT_FILE = "submission.csv"

BATCH_SIZE = 32
EPOCHS = 1000
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
PATIENCE = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data():
    train_df = pd.read_csv(TRAIN_FILE, names=["x", "y", "z"])
    test_df = pd.read_csv(TEST_FILE, names=["x", "y"])

    X = train_df[["x", "y"]].values.astype(np.float32)
    y = train_df["z"].values.astype(np.float32).reshape(-1, 1)
    X_test = test_df.values.astype(np.float32)

    return X, y, X_test, test_df


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


def train_model(model, train_loader, val_loader):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}, Validation MSE: {val_loss:.6f}")


def main():
    X, y, X_test, test_df = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_val = torch.tensor(X_val)
    y_val = torch.tensor(y_val)
    X_test = torch.tensor(X_test)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE
    )

    model = MLP().to(DEVICE)
    print("Training model...")
    train_model(model, train_loader, val_loader)

    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    with torch.no_grad():
        predictions = model(X_test.to(DEVICE)).cpu().numpy()

    test_df["z_pred"] = predictions
    test_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
