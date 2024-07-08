# -------------------------------------------------------- #
# -------------------------------------------------------- #
# implementing a graph neural network with an
# attention mechanism from scratch for attributes classification
# -------------------------------------------------------- #
# -------------------------------------------------------- #

import numpy as np
import torch

from scipy.io import loadmat
annots = loadmat('adj_average_final.mat')
mat=annots['adj_average_final']
adj_matrix_0 = annots['adj_average_final']
adj_matrix = annots['adj_average_final']


annots = loadmat('shuffled_input_features.mat')
mat=annots['shuffled_input_features']
shuffled_input_features_0 = annots['shuffled_input_features']
x = torch.tensor(annots['shuffled_input_features'])


annots = loadmat('shuffled_targets.mat')
mat=annots['shuffled_targets']
shuffled_targets_0 = annots['shuffled_targets']
shuffled_targets_0 = np.where(shuffled_targets_0 == 0, 0.05, shuffled_targets_0)
shuffled_targets_0 = np.where(shuffled_targets_0 == 1, 0.9, shuffled_targets_0)
y = torch.tensor(shuffled_targets_0.T)

edge_index = torch.tensor(adj_matrix).nonzero().t().contiguous()

# Convert the adjacency matrix to a sparse format
adj_matrix = torch.tensor(adj_matrix)
# Extract the indices of non-zero elements
indices = torch.where(adj_matrix != 0)
# Extract the non-zero edge weights
edge_weights = adj_matrix[indices[0], indices[1]]




# ---------------------------------------------------------- #
# ---------------------------------------------------------- #
# ---------------------------------------------------------- #

import torch
from torch_geometric.datasets import Planetoid

# Import dataset from PyTorch Geometric
#dataset = Planetoid(root=".", name="CiteSeer")

#xxx = dataset.data.val_mask
#xx = dataset.data.edge_index


# Print information about the dataset
#print(f'Number of nodes: {x.shape[0]}')



from torch_geometric.utils import degree
from collections import Counter



# Get list of degrees for each node
n = len(adj_matrix)
degrees = []
for i in range(n):
    degree = sum(adj_matrix[i]) + sum(adj_matrix[j][i] for j in range(n))
    degrees.append(degree)

# Count the number of nodes for each degree
numbers = Counter(degrees)

import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv


class GCN(torch.nn.Module):
  """Graph Convolutional Network"""

  def __init__(self, dim_in, dim_h1, dim_h2, dim_out):
    super(GCN, self).__init__()
    self.gcn1 = GCNConv(dim_in, dim_h1)
    self.gcn2 = GCNConv(dim_h1, dim_h2)
    self.gcn3 = GCNConv(dim_h2, dim_out)
    self.bn1 = torch.nn.BatchNorm1d(dim_h1)
    self.bn2 = torch.nn.BatchNorm1d(dim_h2)
    #self.dropout = torch.nn.Dropout(0.5)
    #self.optimizer = torch.optim.Adam(self.parameters(),lr=0.05,weight_decay=5e-4)
    self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
    #self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.01, alpha=0.99)


  def forward(self, x,  edge_index, edge_weights):
    x = torch.nn.functional.leaky_relu(self.gcn1(x, edge_index))
    x = self.bn1(x)
    #x = self.dropout(x)
    x = torch.nn.functional.leaky_relu(self.gcn2(x, edge_index))
    x = self.bn2(x)
    #x = self.dropout(x)
    x = torch.nn.functional.leaky_relu(self.gcn3(x, edge_index))
    return x




def accuracy(pred_y, y):
  """Calculate accuracy."""
  return ((pred_y == y).sum() / len(y)).item()




# Create GCN
gcn = GCN(1, 16, 16, 1)


edge_weights = edge_weights.double()
gcn.double()




criterion = torch.nn.MSELoss()  #torch.nn.CrossEntropyLoss()
optimizer = gcn.optimizer
epochs = 0

gcn.train()
y_pred = torch.zeros(len(x[0, :]),1)
loss_pred = torch.zeros(len(x[0, :]),1)
# Initialize variables to store loss and accuracy
train_loss, train_acc = [], []

# Train the model for multiple epochs
for epoch in range(epochs + 1):
  # Set threshold for gradient norm
  threshold = 1e-5

  # Train the model
  while True:
    for batch in range(len(x[0, :])):
      # Get the current batch of data
      x_batch = x[:, batch].reshape([-1, 1]).double()
      y_batch = y[batch].reshape([-1, 1]).double()

      # Forward pass
      h = gcn(x_batch, edge_index, edge_weights)
      x_sum = torch.median(h).reshape([-1, 1])
      loss = criterion(x_sum, y_batch)

      # Backward pass and optimization
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # Store loss and accuracy
      train_loss.append(loss.item())

      y_pred[batch] = x_sum
      loss_pred[batch] = loss

      # Print metrics every 10 epochs
    if (epoch % 10 == 0):
      print(f'Epoch {epoch:>3} | Train Loss: {np.median(train_loss):.3f}')

    if np.median(train_loss) < 0.095:
      print("Median of Loss below threshold. Training stopped.")
      break

    # Check the norm of the gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(gcn.parameters(), threshold)
    if grad_norm < threshold:
      print("Gradient norm below threshold. Training stopped.")
      break


