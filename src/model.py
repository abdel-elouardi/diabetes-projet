import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def prepare_data(df):
    X = df.drop('Outcome', axis=1).values
    y = df['Outcome'].values

    # Normalisation
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Conversion en tenseurs PyTorch
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)

    return X_train, X_test, y_train, y_test, scaler

class DiabeteModel(nn.Module):
    def __init__(self):
        super(DiabeteModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

def train_model(X_train, y_train, epochs=100):
    model = DiabeteModel()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("=== Entraînement du modèle ===")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    return model

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test).squeeze()
        predicted = (outputs >= 0.5).float()
        accuracy = (predicted == y_test).float().mean()
        print(f"\n=== Résultats ===")
        print(f"Précision : {accuracy.item() * 100:.2f}%")

def save_model(model, path="models/model.pt"):
    torch.save(model, path)
    print(f"✅ Modèle sauvegardé dans {path}")