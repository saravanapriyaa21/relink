import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print("🚀 Training ReLink GNN on Real NCRB + Census Data...")

# Load preprocessed dataset
df = pd.read_csv("../data/district_risk_2022.csv")

# Use the main feature columns
feature_cols = ["missing_rate", "female_ratio", "literacy_rate", "workers_ratio"]
x = torch.tensor(df[feature_cols].values, dtype=torch.float32)

# Create synthetic edges (approx adjacency)
num_nodes = len(df)
edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))

# Zone classification (based on risk score)
def classify_zone(score):
    if score < 0.35:
        return 0
    elif score < 0.65:
        return 1
    else:
        return 2

df["zone_label"] = df["risk_score"].apply(classify_zone)
y = torch.tensor(df["zone_label"].values, dtype=torch.long)

# Split for training/testing
train_idx, test_idx = train_test_split(np.arange(num_nodes), test_size=0.2, random_state=42)

# Define GNN
class ReLinkGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 3)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return torch.log_softmax(x, dim=1)

# Prepare data
data = Data(x=x, edge_index=edge_index, y=y)

# Train model
model = ReLinkGNN(in_channels=x.size(1), hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

print("🧠 Training started...")
for epoch in range(201):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        model.eval()
        pred = out.argmax(dim=1)
        acc = int((pred[test_idx] == data.y[test_idx]).sum()) / len(test_idx)
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Accuracy: {acc:.3f}")

# Save model
torch.save(model.state_dict(), "../models/relink_gnn_real.pt")
print("✅ Model trained & saved to ../models/relink_gnn_real.pt")
