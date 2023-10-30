from torch_geometric.data import Data
import torch

def convert_to_torch_geometric_data(graph, label):
    # Convert to PyTorch tensors
    nodes = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    x = torch.tensor([graph.nodes[node]['feature'] for node in nodes], dtype=torch.float)
    
    # Create edge_index tensor with mapped indices
    edge_list = [(node_to_idx[src], node_to_idx[dst]) for src, dst in graph.edges()]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    y = torch.tensor([label], dtype=torch.long)
    
    # Create a PyTorch Geometric data object
    return Data(x=x, edge_index=edge_index, y=y)

# Create a list of PyTorch Geometric data objects
data_list = [convert_to_torch_geometric_data(graph, 1 if label == 'spam' else 0) for graph, label in zip(df['Graph'], df['Category'])]
