import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv, FAConv
from torch.nn import Parameter
import numpy as np, itertools, random, copy, math
import math
import scipy.sparse as sp
import ipdb
# from HypergraphConv import HypergraphConv
from torch_geometric.nn import GCNConv
from itertools import permutations
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import add_self_loops
from graphgcn import GraphGCN


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x, dia_len):
        """
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        tmpx = torch.zeros(0).cuda()
        tmp = 0
        for i in dia_len:
            a = x[tmp:tmp+i].unsqueeze(1)
            a = a + self.pe[:a.size(0)]
            tmpx = torch.cat([tmpx,a], dim=0)
            tmp = tmp+i
        #x = x + self.pe[:x.size(0)]
        tmpx = tmpx.squeeze(1)
        return self.dropout(tmpx)

class GCN(nn.Module):
    def __init__(self, n_dim, nhidden, dropout, lamda, alpha, variant, return_feature, use_residue, 
                new_graph='full',n_speakers=2, modals=['a','v','l'], use_speaker=True, use_modal=False, num_L=3, num_K=4, original_gcn=False, graph_masking=True):
        super(GCN, self).__init__()
        self.return_feature = return_feature  #True
        self.use_residue = use_residue
        self.new_graph = new_graph

        self.original_gcn = original_gcn
        self.graph_masking = graph_masking
        
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.modals = modals
        self.modal_embeddings = nn.Embedding(3, n_dim)
        self.speaker_embeddings = nn.Embedding(n_speakers, n_dim)
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.use_position = False
        #------------------------------------    
        self.fc1 = nn.Linear(n_dim, nhidden)         
        self.num_L =  num_L
        self.num_K =  num_K
        self.act_fn = nn.ReLU()
        self.hyperedge_weight = nn.Parameter(torch.ones(1000))
        self.EW_weight = nn.Parameter(torch.ones(5200))
        self.hyperedge_attr1 = nn.Parameter(torch.rand(nhidden))
        self.hyperedge_attr2 = nn.Parameter(torch.rand(nhidden))
        for kk in range(num_K):
            setattr(self,'conv%d' %(kk+1), GraphGCN(nhidden, nhidden, original_gcn = self.original_gcn, graph_masking=self.graph_masking))

    def forward(self, a, v, l, dia_len, qmask, epoch):
        qmask = torch.cat([qmask[:x,i,:] for i,x in enumerate(dia_len)],dim=0)
        spk_idx = torch.argmax(qmask, dim=-1)
        spk_emb_vector = self.speaker_embeddings(spk_idx)
        if self.use_speaker:
            if 'l' in self.modals:
                l += spk_emb_vector
        if self.use_position:
            if 'l' in self.modals:
                l = self.l_pos(l, dia_len)
            if 'a' in self.modals:
                a = self.a_pos(a, dia_len)
            if 'v' in self.modals:
                v = self.v_pos(v, dia_len)
        if self.use_modal:  
            emb_idx = torch.LongTensor([0, 1, 2]).to("cuda:3")
            emb_vector = self.modal_embeddings(emb_idx)

            if 'a' in self.modals:
                a += emb_vector[0].reshape(1, -1).expand(a.shape[0], a.shape[1])
            if 'v' in self.modals:
                v += emb_vector[1].reshape(1, -1).expand(v.shape[0], v.shape[1])
            if 'l' in self.modals:
                l += emb_vector[2].reshape(1, -1).expand(l.shape[0], l.shape[1])                                  

        #---------------------------------------
        
        
        
        gnn_edge_index, gnn_features = self.create_gnn_index(a, v, l, dia_len, self.modals)
        x1 = self.fc1(gnn_features)  
        out = x1
        gnn_out = x1
        for kk in range(self.num_K):
            if self.original_gcn:
                gnn_out = getattr(self,'conv%d' %(kk+1))(gnn_out,gnn_edge_index)
            else:
                gnn_out = gnn_out + getattr(self,'conv%d' %(kk+1))(gnn_out,gnn_edge_index)

        out2 = torch.cat([out,gnn_out], dim=1)
        if self.use_residue:
            out2 = torch.cat([gnn_features, out2], dim=-1)
        out1 = self.reverse_features(dia_len, out2)
        #---------------------------------------
        return out1

    def reverse_features(self, dia_len, features):
        l=[]
        a=[]
        v=[]
        for i in dia_len:
            ll = features[0:1*i]
            aa = features[1*i:2*i]
            vv = features[2*i:3*i]
            features = features[3*i:]
            l.append(ll)
            a.append(aa)
            v.append(vv)
        tmpl = torch.cat(l,dim=0)
        tmpa = torch.cat(a,dim=0)
        tmpv = torch.cat(v,dim=0)
        features = torch.cat([tmpl, tmpa, tmpv], dim=-1)
        return features


    def create_gnn_index(self, a, v, l, dia_len, modals):
        num_modality = len(modals)
        node_count = 0
        index =[]
        tmp = []
        
        
        for i in dia_len:
            nodes = list(range(i*num_modality))
            
            nodes = [j + node_count for j in nodes] 
            
            nodes_l = nodes[0:i*num_modality//3]
            nodes_a = nodes[i*num_modality//3:i*num_modality*2//3]
            nodes_v = nodes[i*num_modality*2//3:]
            index = index + list(permutations(nodes_l,2)) + list(permutations(nodes_a,2)) + list(permutations(nodes_v,2))
            
            Gnodes=[]
            for _ in range(i):
                Gnodes.append([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]])
                
            for ii, _ in enumerate(Gnodes):
                tmp = tmp +  list(permutations(_,2))
                
            if node_count == 0:
                ll = l[0:0+i]
                
                aa = a[0:0+i]
                vv = v[0:0+i]
                features = torch.cat([ll,aa,vv],dim=0)
                temp = 0+i
            else:
                
                ll = l[temp:temp+i]
                aa = a[temp:temp+i]
                vv = v[temp:temp+i]
                features_temp = torch.cat([ll,aa,vv],dim=0)
                features =  torch.cat([features,features_temp],dim=0)
                temp = temp+i
            node_count = node_count + i*num_modality
        edge_index = torch.cat([torch.LongTensor(index).T,torch.LongTensor(tmp).T],1).to("cuda:0")
        
        return edge_index, features
    
    
    
class UNIMODALGCN(nn.Module):
    def __init__(self, n_dim, nhidden, dropout, lamda, alpha, return_feature, use_residue, 
                new_graph='full',n_speakers=2, modality=None, use_speaker=True, use_modal=False, num_L=3, num_K=4):
        super(UNIMODALGCN, self).__init__()
        self.return_feature = return_feature  #True
        self.use_residue = use_residue
        self.new_graph = new_graph

        self.modality = modality
        self.speaker_embeddings = nn.Embedding(n_speakers, n_dim)
        self.use_speaker = use_speaker
        # self.use_position = False
        #------------------------------------    
        self.fc1 = nn.Linear(n_dim, nhidden)         
        self.num_K =  num_K

        for kk in range(num_K):
            setattr(self,'conv%d' %(kk+1), GraphGCN(nhidden, nhidden))

    def forward(self,uni_feature, dia_len, qmask, epoch):
        qmask = torch.cat([qmask[:x,i,:] for i,x in enumerate(dia_len)],dim=0)
        spk_idx = torch.argmax(qmask, dim=-1)
        spk_emb_vector = self.speaker_embeddings(spk_idx)
        if self.use_speaker:
            if self.modality == "text":
                uni_feature += spk_emb_vector
        # if self.use_position:
        #     if 'l' in self.modals:
        #         l = self.l_pos(l, dia_len)
        #     if 'a' in self.modals:
        #         a = self.a_pos(a, dia_len)
        #     if 'v' in self.modals:
        #         v = self.v_pos(v, dia_len)
        # if self.use_modal:  
        #     emb_idx = torch.LongTensor([0, 1, 2]).to("cuda:3")
        #     emb_vector = self.modal_embeddings(emb_idx)

        #     if 'a' in self.modals:
        #         a += emb_vector[0].reshape(1, -1).expand(a.shape[0], a.shape[1])
        #     if 'v' in self.modals:
        #         v += emb_vector[1].reshape(1, -1).expand(v.shape[0], v.shape[1])
        #     if 'l' in self.modals:
        #         l += emb_vector[2].reshape(1, -1).expand(l.shape[0], l.shape[1])                                  

        #---------------------------------------
        
        
        
        gnn_edge_index, gnn_features = self.create_gnn_index(uni_feature, dia_len)
        x1 = self.fc1(gnn_features)  
        out = x1
        gnn_out = x1
        for kk in range(self.num_K):
            gnn_out = gnn_out + getattr(self,'conv%d' %(kk+1))(gnn_out,gnn_edge_index)

        out2 = torch.cat([out,gnn_out], dim=1)
        if self.use_residue:
            out2 = torch.cat([gnn_features, out2], dim=-1)
        out1 = self.reverse_features(dia_len, out2)
        #---------------------------------------
        return out1

    def reverse_features(self, dia_len, features):
        tmplist = []
        
        for i in dia_len:
            tmpfeatures = features[0:1*i]
            features = features[1*i:]
            tmplist.append(tmpfeatures)
        features = torch.cat(tmplist,dim=0)
        return features


    def create_gnn_index(self, uni_feature, dia_len):
        # num_modality = len(modals)
        node_count = 0
        index =[]
        tmp = []
        
        
        for i in dia_len:
            nodes = list(range(i))
            
            nodes = [j + node_count for j in nodes] 
            
            index = index + list(permutations(nodes,2)) 
            
            if node_count == 0:
                features = uni_feature[0:0+i]
                temp = 0+i
            else:
                features_temp = uni_feature[temp:temp+i]                
                features =  torch.cat([features,features_temp],dim=0)
                temp = temp+i
            node_count = node_count + i
        edge_index = torch.LongTensor(index).T.to("cuda:0")
        
        return edge_index, features
