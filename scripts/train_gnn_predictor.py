import torch
import pandas as pd
import wandb
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from optimol.models.predictor_gnn import Keap1PredictorGNN
from optimol.utils.graph_utils import create_pytorch_geom_dataset
from sklearn.model_selection import KFold
from loguru import logger

def train():
    # Инициализация W&B
    wandb.init(project="optimol-keap1", name="gnn-baseline")
    
    # Загрузка данных
    df = pd.read_csv("data/processed/keap1_cleaned.csv")
    dataset = create_pytorch_geom_dataset(df)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        logger.info(f"--- Training Fold {fold} ---")
        
        train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=32, shuffle=True)
        val_loader = DataLoader([dataset[i] for i in val_idx], batch_size=32)
        
        model = Keap1PredictorGNN(node_features=6).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        for epoch in range(50): # Для теста поставим 50 эпох
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out.view(-1), batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                model.eval()
                val_mse = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        pred = model(batch)
                        val_mse += criterion(pred.view(-1), batch.y).item()
                
                logger.info(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | Val RMSE: {(val_mse/len(val_loader))**0.5:.4f}")
                wandb.log({f"fold_{fold}_val_rmse": (val_mse/len(val_loader))**0.5})

            torch.save(model.state_dict(), "models/checkpoints/predictor_gnn_final.pt")
            logger.success("Predictor weights saved!")

    # В конце функции train() после цикла KFold:
    logger.success("Training complete!")
    # Сохраняем модель из последнего фолда (или можно выбрать лучшую)
    os.makedirs("models/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "models/checkpoints/predictor_gnn_final.pt")
    logger.success("Predictor weights saved to models/checkpoints/predictor_gnn_final.pt")

if __name__ == "__main__":
    train()
