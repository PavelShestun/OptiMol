import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from optimol.models.kan_layer import KAN

class Keap1PredictorGNN(torch.nn.Module):
    def __init__(self, node_features=6, hidden_dim=64):
        super(Keap1PredictorGNN, self).__init__()
        
        # Графовая часть (GAT v2 для SOTA точности)
        self.conv1 = GATConv(node_features, hidden_dim, heads=4, dropout=0.1)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=0.1)
        
        # Наш KAN
        self.kan = KAN([hidden_dim * 4, 32, 1]) 

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 1. Извлечение признаков графа
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        
        # 2. Агрегация атомов в вектор молекулы
        x = global_mean_pool(x, batch)
        
        # 3. Предсказание через KAN
        return self.kan(x)
