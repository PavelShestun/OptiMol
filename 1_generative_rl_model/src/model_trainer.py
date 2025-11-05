# src/model_trainer.py
import pandas as pd
import numpy as np
import joblib
import logging
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from .config import MORGAN_GENERATOR, KEAP1_MODEL_PATH, EGFR_MODEL_PATH, IKKB_MODEL_PATH
from .property_calculator import get_features
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchkan import KAN

logger = logging.getLogger(__name__)

class KANRegressor(nn.Module):
    def __init__(self, input_dim):
        super(KANRegressor, self).__init__()
        self.kan = KAN(layers_hidden=[input_dim, 64, 1], grid_size=5, spline_order=3)

    def forward(self, x):
        return self.kan(x)

    def predict(self, X):
        device = next(self.parameters()).device
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        self.eval()
        with torch.no_grad():
            return self.kan(X_tensor).cpu().numpy()

def train_kan_regressor(X, y):
    """Trains a KAN regressor model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X.shape[1]
    model = KANRegressor(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(50):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    return model

def train_and_select_best_model(df: pd.DataFrame, model_filename: str):
    """Обучает несколько моделей, выбирает лучшую и сохраняет ее."""
    if df is None or df.empty or len(df) < 50:
        logger.warning(f"Недостаточно данных для обучения модели {model_filename} ({len(df) if df is not None else 0} строк).")
        return None

    logger.info(f"--- Training KAN model for [{model_filename}] ---")

    mols = [Chem.MolFromSmiles(smi) for smi in df['canonical_smiles']]
    valid_indices = [i for i, m in enumerate(mols) if m is not None]

    mols_valid = [mols[i] for i in valid_indices]
    df_valid = df.iloc[valid_indices]

    if len(df_valid) < 50:
        logger.warning(f"After SMILES validation, not enough data remains ({len(df_valid)}), training for {model_filename} skipped.")
        return None

    y = df_valid['pIC50'].values
    try:
        X = np.vstack([get_features(m) for m in mols_valid])
    except Exception as e:
        logger.error(f"Error generating features for {model_filename}: {e}")
        return None

    mask = np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    if len(X) < 50:
        logger.warning(f"After cleaning features, not enough data remains ({len(X)}), training for {model_filename} skipped.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        logger.info(f"  [*] Training KAN...")
        model = train_kan_regressor(X_train, y_train)
        score = r2_score(y_test, model.predict(X_test))
        logger.info(f"    -> R2-score on test set: {score:.3f}")

        joblib.dump(model, model_filename)
        logger.info(f"  [+] KAN model saved to: {model_filename}")
        return model
    except Exception as e:
        logger.error(f"    -> Error training KAN: {e}")
        return None

def prepare_models(keap1_clean, egfr_clean, ikkb_clean):
    """Главная функция для подготовки всех моделей-предикторов."""
    logger.info("\n[*] Этап 3: Конкурс моделей-предикторов...")
    
    keap1_model = train_and_select_best_model(keap1_clean, KEAP1_MODEL_PATH)
    egfr_model = train_and_select_best_model(egfr_clean, EGFR_MODEL_PATH)
    ikkb_model = train_and_select_best_model(ikkb_clean, IKKB_MODEL_PATH)
    
    logger.info("\n✅ Все лучшие модели-предикторы готовы.")
    return keap1_model, egfr_model, ikkb_model
