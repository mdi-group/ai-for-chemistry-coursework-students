import numpy as np
import torch
import torch.nn as nn
from src.utils_gnn import MaterialsDataset, collate_fn, plot_sample, scatter_sum, scatter_mean

class GraphConvolution(nn.Module):

    def __init__(self, node_feat_dim, edge_feat_dim):
        """
        Convolutional layer for graphs.

        Parameters
        ----------
        node_feat_dim : int
          Number of node features.
        edge_feat_dim : int
          Number of edge features.
        """
        super().__init__()

        # linear layer used for the gated MLP
        self.lin1 = nn.Linear(
            2 * node_feat_dim + edge_feat_dim,
            2 * node_feat_dim,
        )

        # normalisation layers
        self.bn1 = nn.BatchNorm1d(2 * node_feat_dim)
        self.bn2 = nn.BatchNorm1d(node_feat_dim)

    def forward(self, node_feat, edge_feat, edge_src, edge_dst):
        """Perform the convolution.

        Parameters
        ----------
        node_feat : Tensor
            The node features.
        edge_feat : Tensor
            The edge features
        edge_src : Tensor
            The indices of the central nodes for the edges.
        edge_dst : Tensor
            The indices of the desintation nodes for the edges.
        """
        # concatenate node and edge features
        m = torch.cat([node_feat[edge_src], node_feat[edge_dst], edge_feat], dim=1)

        # gated MLP
        z = self.lin1(m)
        z = self.bn1(z)
        z1, z2 = z.chunk(2, dim=1)
        z1 = nn.Sigmoid()(z1)
        z2 = nn.Softplus()(z2)
        z = z1 * z2

        # pool features
        z = scatter_sum(z, edge_src, dim=0, dim_size=node_feat.shape[0])

        # pass through normalisation layer
        return nn.Softplus()(self.bn2(z) + node_feat)

class CGCNN(nn.Module):
    def __init__(
        self,
        node_feat_dim,
        edge_feat_dim,
        node_hidden_dim=64,
        num_graph_conv_layers=3,
        fc_feat_dim=128
    ):
        """
        Crystal Graph Convolutional Neural Network 

        Parameters
        ----------
        node_feat_dim : int
          Number of initial node features from one-hot encoding.
        edge_feat_dim : int
          Number of bond features.
        node_hidden_dim : int
          The number of features in the node embedding.
        num_graph_conv_layers: int
          Number of convolutional layers.
        fc_feat_dim: int
          Number of hidden features after pooling.
        """
        super().__init__()

        # dense layer to transform one-hot encoded node features to embedding
        self.embedding = nn.Linear(node_feat_dim, node_hidden_dim)

        # set up the convolutions
        convs = []
        for _ in range(num_graph_conv_layers):
            convs.append(GraphConvolution(node_feat_dim=node_hidden_dim, edge_feat_dim=edge_feat_dim))
        self.convs = nn.ModuleList(convs)

        # dense layer to turn final node embeddings to the crystal features
        self.conv_to_fc = nn.Sequential(
           nn.Linear(node_hidden_dim, fc_feat_dim), nn.Softplus()
        )

        # dense layer to get the final target value
        self.fc_out = nn.Linear(fc_feat_dim, 1)

    def forward(self, batch):
        """
        Predict the target property given a batch of data.

        Parameters
        ----------
        batch : Batch
            The data to pass through the network.
        """
        # get initial node embedding
        node_feat = self.embedding(batch.node_feat)

        # apply convolutions
        for conv_func in self.convs:
            node_feat = conv_func(node_feat, batch.edge_feat, batch.edge_src, batch.edge_dst)

        # pool node vectors
        crys_feat = scatter_mean(node_feat, batch.batch, dim=0, dim_size=batch.batch.max() + 1)

        # pass pooled vector through FC layer with activation
        crys_feat = self.conv_to_fc(crys_feat)

        # pass crystal features through final fully-connected layer
        return self.fc_out(crys_feat)
