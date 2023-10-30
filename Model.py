import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import torch.optim as optim

# Create a DataLoader for the dataset
#loader = DataLoader(data_list, batch_size=32, shuffle=True,drop_last=True)
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(len(unique_words), 16)
        self.conv2 = GCNConv(16, 16)
        self.classifier = torch.nn.Linear(16, 2)  # 2 classes: Spam or Ham

    def forward(self, data, batch = None):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        if data.num_edges == 0 or data.num_nodes == 0:
        # Handle empty graphs, e.g., return a zero tensor
            return torch.zeros((1, self.num_classes))
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        print("before global pooling",batch.shape)
        # Global mean pooling
        x = global_mean_pool(x, batch)
        print("after global pooling",batch.shape)
        # Classifier
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)
