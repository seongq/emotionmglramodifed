import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.utils import degree, add_self_loops

def generate_random_mask(num_nodes, num_features, mask_prob=0.5):
    """
    随机生成掩码矩阵。根据给定概率 mask_prob 为每个特征设置掩码。
    :param num_nodes: 节点数量
    :param num_features: 每个节点的特征维度
    :param mask_prob: 掩码概率（0-1之间），越高表示更多特征会被屏蔽。
    :return: 随机掩码矩阵
    """
    mask = torch.rand((num_nodes, num_features)) > mask_prob  # 生成布尔掩码
    return mask

class GraphGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add', original_gcn=False, graph_masking=True):
        super(GraphGCN, self).__init__(aggr='add')  # "Add" aggregation.
        #self.lin = torch.nn.Linear(in_channels, out_channels)
        self.original_gcn = original_gcn
        self.graph_masking = graph_masking
        if original_gcn:
            self.linear = torch.nn.Linear(in_channels, out_channels)
        else:
            self.gate = torch.nn.Linear(2*in_channels, 1)
    def forward(self, x, edge_index):
        num_nodes, dim = x.shape
        if self.graph_masking:
            mask = generate_random_mask(num_nodes, dim, mask_prob=0.5).to(x.device)
            x = x * mask

        # For original GCN, use A+I
        if self.original_gcn:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if self.original_gcn:
            x = self.linear(x)

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j, edge_index, size):
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        if self.original_gcn:
            return norm.view(-1, 1) * x_j
        else:
            h2 = torch.cat([x_i, x_j], dim=1)
            alpha_g = torch.tanh(self.gate(h2))
            return norm.view(-1, 1) * x_j * alpha_g

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out