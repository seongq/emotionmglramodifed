import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import  GraphConv
import numpy as np, itertools, random, copy, math
from model_gcn import GCN, UNIMODALGCN

def print_grad(grad):
    print('the grad is', grad[2][0:5])
    return grad

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2.5, alpha = 1, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
    
    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        labels_length = logits.size(1)
        seq_length = logits.size(0)

        new_label = labels.unsqueeze(1)
        # print(seq_length)
        # print(labels_length)
        label_onehot = torch.zeros([seq_length, labels_length]).cuda().scatter_(1, new_label, 1)

        log_p = F.log_softmax(logits,-1)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()

class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1,1)
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        return loss


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len
        target -> batch*seq_len
        mask -> batch*seq_len
        """
        loss = self.loss(pred*mask, target)/torch.sum(mask)
        return loss


class UnMaskedWeightedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(UnMaskedWeightedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        """
        if type(self.weight)==type(None):
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred, target)\
                            /torch.sum(self.weight[target])
        return loss


class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim,1,bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M)
        alpha = F.softmax(scale, dim=0).permute(1,2,0)
        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:]
        return attn_pool, alpha


class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type!='concat' or alpha_dim!=None
        assert att_type!='dot' or mem_dim==cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type=='general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type=='general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
        elif att_type=='concat':
            self.transform = nn.Linear(cand_dim+mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim) cand_dim == mem_dim?
        mask -> (batch, seq_len)
        """
        if type(mask)==type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type=='dot':
            M_ = M.permute(1,2,0)
            x_ = x.unsqueeze(1)
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)
        elif self.att_type=='general':
            M_ = M.permute(1,2,0)
            x_ = self.transform(x).unsqueeze(1)
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)
        elif self.att_type=='general2':
            M_ = M.permute(1,2,0)
            x_ = self.transform(x).unsqueeze(1)
            mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2)
            M_ = M_ * mask_
            alpha_ = torch.bmm(x_, M_)*mask.unsqueeze(1)
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2)
            alpha_masked = alpha_*mask.unsqueeze(1)
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)
            alpha = alpha_masked/alpha_sum
        else:
            M_ = M.transpose(0,1)
            x_ = x.unsqueeze(1).expand(-1,M.size()[0],-1)
            M_x_ = torch.cat([M_,x_],2)
            mx_a = F.tanh(self.transform(M_x_))
            alpha = F.softmax(self.vector_prod(mx_a),1).transpose(1,2)

        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:]
        return attn_pool, alpha



class GRUModel(nn.Module):

    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5):
        
        super(GRUModel, self).__init__()
        
        self.n_classes = n_classes
        self.dropout   = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        self.linear = nn.Linear(2*D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)
        
    def forward(self, U, qmask, umask, att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions, hidden = self.gru(U)
        alpha, alpha_f, alpha_b = [], [], []
        
        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions,t,mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob, alpha, alpha_f, alpha_b, emotions


class LSTMModel(nn.Module):

    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5):
        
        super(LSTMModel, self).__init__()
        
        self.n_classes = n_classes
        self.dropout   = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        self.linear = nn.Linear(2*D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def forward(self, U, qmask, umask, att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions, hidden = self.lstm(U)
        alpha, alpha_f, alpha_b = [], [], []
        
        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions,t,mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob, alpha, alpha_f, alpha_b, emotions


class MaskedEdgeAttention(nn.Module):

    def __init__(self, input_dim, max_seq_len, no_cuda):
        """
        Method to compute the edge weights, as in Equation 1. in the paper. 
        attn_type = 'attn1' refers to the equation in the paper.
        For slightly different attention mechanisms refer to attn_type = 'attn2' or attn_type = 'attn3'
        """

        super(MaskedEdgeAttention, self).__init__()
        
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.scalar = nn.Linear(self.input_dim, self.max_seq_len, bias=False)
        self.matchatt = MatchingAttention(self.input_dim, self.input_dim, att_type='general2')
        self.simpleatt = SimpleAttention(self.input_dim)
        self.att = Attention(self.input_dim, score_function='mlp')
        self.no_cuda = no_cuda

    def forward(self, M, lengths, edge_ind):
        """
        M -> (seq_len, batch, vector)
        lengths -> length of the sequences in the batch
        edge_idn 
        """
        attn_type = 'attn1'

        if attn_type == 'attn1':

            scale = self.scalar(M)
            alpha = F.softmax(scale, dim=0).permute(1, 2, 0)
            if not self.no_cuda:
                mask = Variable(torch.ones(alpha.size()) * 1e-10).detach().cuda()
                mask_copy = Variable(torch.zeros(alpha.size())).detach().cuda()
                
            else:
                mask = Variable(torch.ones(alpha.size()) * 1e-10).detach()
                mask_copy = Variable(torch.zeros(alpha.size())).detach()
            
            edge_ind_ = []
            for i, j in enumerate(edge_ind):
                for x in j:
                    edge_ind_.append([i, x[0], x[1]])

            edge_ind_ = np.array(edge_ind_).transpose()
            mask[edge_ind_] = 1
            mask_copy[edge_ind_] = 1
            masked_alpha = alpha * mask
            _sums = masked_alpha.sum(-1, keepdim=True)
            scores = masked_alpha.div(_sums) * mask_copy

            return scores

        elif attn_type == 'attn2':
            scores = torch.zeros(M.size(1), self.max_seq_len, self.max_seq_len, requires_grad=True)

            if not self.no_cuda:
                scores = scores.cuda()


            for j in range(M.size(1)):
            
                ei = np.array(edge_ind[j])

                for node in range(lengths[j]):
                
                    neighbour = ei[ei[:, 0] == node, 1]

                    M_ = M[neighbour, j, :].unsqueeze(1)
                    t = M[node, j, :].unsqueeze(0)
                    _, alpha_ = self.simpleatt(M_, t)
                    scores[j, node, neighbour] = alpha_

        elif attn_type == 'attn3':
            scores = torch.zeros(M.size(1), self.max_seq_len, self.max_seq_len, requires_grad=True)

            if not self.no_cuda:
                scores = scores.cuda()

            for j in range(M.size(1)):

                ei = np.array(edge_ind[j])

                for node in range(lengths[j]):

                    neighbour = ei[ei[:, 0] == node, 1]

                    M_ = M[neighbour, j, :].unsqueeze(1).transpose(0, 1)
                    t = M[node, j, :].unsqueeze(0).unsqueeze(0).repeat(len(neighbour), 1, 1).transpose(0, 1)
                    _, alpha_ = self.att(M_, t)
                    scores[j, node, neighbour] = alpha_[0, :, 0]

        return scores


def simple_batch_graphify(features, lengths, no_cuda):
    node_features = []
    
    batch_size = features.size(1)
    
    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])
    
    # for x in node_features:
        
    node_features = torch.cat(node_features, dim=0)  

    if not no_cuda:
        node_features = node_features.to("cuda:0")
    
    return node_features



class MMGatedAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, att_type='general'):
        super(MMGatedAttention, self).__init__()
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        self.dropouta = nn.Dropout(0.5)
        self.dropoutv = nn.Dropout(0.5)
        self.dropoutl = nn.Dropout(0.5)
        if att_type=='av_bg_fusion':
            self.transform_al = nn.Linear(mem_dim*2, cand_dim, bias=True)
            self.scalar_al = nn.Linear(mem_dim, cand_dim)
            self.transform_vl = nn.Linear(mem_dim*2, cand_dim, bias=True)
            self.scalar_vl = nn.Linear(mem_dim, cand_dim)
        elif att_type=='general':
            self.transform_l = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_v = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_a = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_av = nn.Linear(mem_dim*3,1)
            self.transform_al = nn.Linear(mem_dim*3,1)
            self.transform_vl = nn.Linear(mem_dim*3,1)

    def forward(self, a, v, l, modals=None):
        a = self.dropouta(a) if len(a) !=0 else a
        v = self.dropoutv(v) if len(v) !=0 else v
        l = self.dropoutl(l) if len(l) !=0 else l
        if self.att_type == 'av_bg_fusion':
            if 'a' in modals:
                fal = torch.cat([a, l],dim=-1)
                Wa = torch.sigmoid(self.transform_al(fal))
                hma = Wa*(self.scalar_al(a))
            if 'v' in modals:
                fvl = torch.cat([v, l],dim=-1)
                Wv = torch.sigmoid(self.transform_vl(fvl))
                hmv = Wv*(self.scalar_vl(v))
            if len(modals) == 3:
                hmf = torch.cat([l,hma,hmv], dim=-1)
            elif 'a' in modals:
                hmf = torch.cat([l,hma], dim=-1)
            elif 'v' in modals:
                hmf = torch.cat([l,hmv], dim=-1)
            return hmf
        elif self.att_type == 'general':
            ha = torch.tanh(self.transform_a(a)) if 'a' in modals else a
            hv = torch.tanh(self.transform_v(v)) if 'v' in modals else v
            hl = torch.tanh(self.transform_l(l)) if 'l' in modals else l

            if 'a' in modals and 'v' in modals:
                z_av = torch.sigmoid(self.transform_av(torch.cat([a,v,a*v],dim=-1)))
                h_av = z_av*ha + (1-z_av)*hv
                if 'l' not in modals:
                    return h_av
            if 'a' in modals and 'l' in modals:
                z_al = torch.sigmoid(self.transform_al(torch.cat([a,l,a*l],dim=-1)))
                h_al = z_al*ha + (1-z_al)*hl
                if 'v' not in modals:
                    return h_al
            if 'v' in modals and 'l' in modals:
                z_vl = torch.sigmoid(self.transform_vl(torch.cat([v,l,v*l],dim=-1)))
                h_vl = z_vl*hv + (1-z_vl)*hl
                if 'a' not in modals:
                    return h_vl
            return torch.cat([h_av, h_al, h_vl],dim=-1)
        
class MultiHeadCrossModalAttention(nn.Module):
    def __init__(self, img_dim, txt_dim, hidden_dim, num_heads):
        super(MultiHeadCrossModalAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 多头投影层，将图像和文本分别映射到多头的 Q、K、V
        self.img_query_proj = nn.Linear(img_dim, hidden_dim)
        self.txt_key_proj = nn.Linear(txt_dim, hidden_dim)
        self.txt_value_proj = nn.Linear(txt_dim, hidden_dim)

        # 最后的输出线性变换
        self.output_proj = nn.Linear(hidden_dim, img_dim)

    def forward(self, img_features, txt_features):
        """
        :param img_features: [batch_size, num_regions, img_dim]
        :param txt_features: [batch_size, num_words, txt_dim]
        :return: 融合后的特征
        """
        B, R, _ = img_features.shape  # B: batch_size, R: num_regions
        _, W, _ = txt_features.shape  # W: num_words

        # 线性投影得到 Q、K、V，并 reshape 为多头格式
        Q = self.img_query_proj(img_features).view(B, R, self.num_heads, self.head_dim).transpose(1, 2)  
        K = self.txt_key_proj(txt_features).view(B, W, self.num_heads, self.head_dim).transpose(1, 2)  
        V = self.txt_value_proj(txt_features).view(B, W, self.num_heads, self.head_dim).transpose(1, 2)  

        # 计算注意力权重: Q·K^T / sqrt(d_k)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  
        attention_weights = F.softmax(attention_scores, dim=-1)  

        # 加权求和得到上下文表示
        attended_features = torch.matmul(attention_weights, V)  

        # 合并多头的结果
        attended_features = attended_features.transpose(1, 2).contiguous().view(B, R, -1)  

        # 输出线性变换
        output = img_features + self.output_proj(attended_features)  
        return output


class Model(nn.Module):

    def __init__(self, base_model, D_m, D_g, D_e, graph_hidden_size, n_speakers,
                 n_classes=7, dropout=0.5, avec=False, 
                 no_cuda=False, graph_type='relation', use_topic=False, alpha=0.2, multiheads=6, graph_construct='direct', use_GCN=False,use_residue=True,
                 dynamic_edge_w=False,D_m_v=512,D_m_a=100,modals='avl',att_type='gated',av_using_lstm=False, dataset='IEMOCAP',
                 use_speaker=True, use_modal=False, num_L = 3, num_K = 4, modality_self_attention=False, modality_LSTM=False, original_gcn=False, graph_masking=True):
        
        super(Model, self).__init__()
        self.original_gcn = original_gcn
        self.graph_masking = graph_masking
        self.modality_self_attention = modality_self_attention
        self.modality_LSTM = modality_LSTM
        self.base_model = base_model
        self.avec = avec
        self.no_cuda = no_cuda
        self.graph_type=graph_type
        self.alpha = alpha
        self.multiheads = multiheads
        self.graph_construct = graph_construct
        self.use_topic = use_topic
        self.dropout = dropout
        self.use_GCN = use_GCN
        self.use_residue = use_residue
        self.dynamic_edge_w = dynamic_edge_w
        self.return_feature = True
        self.modals = [x for x in modals]  # a, v, l
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.att_type = att_type
        self.normBNa = nn.BatchNorm1d(1024, affine=True)
        self.normBNb = nn.BatchNorm1d(1024, affine=True)
        self.normBNc = nn.BatchNorm1d(1024, affine=True)
        self.normBNd = nn.BatchNorm1d(1024, affine=True)

        if self.att_type == 'gated' or self.att_type == 'concat_subsequently' or self.att_type == 'concat_DHT':
            self.multi_modal = True
            self.av_using_lstm = av_using_lstm
        else:
            self.multi_modal = False
        self.dataset = dataset

        if self.base_model == 'LSTM':
            if not self.multi_modal:
                if len(self.modals) == 3:
                    hidden_ = 250
                elif ''.join(self.modals) == 'al':
                    hidden_ = 150
                elif ''.join(self.modals) == 'vl':
                    hidden_ = 150
                else:
                    hidden_ = 100
                self.linear_ = nn.Linear(D_m, hidden_)
                self.lstm = nn.LSTM(input_size=hidden_, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            else:
                if 'a' in self.modals:
                    hidden_a = D_g
                    self.linear_a = nn.Linear(D_m_a, hidden_a)
                    if self.av_using_lstm:
                        self.lstm_a = nn.LSTM(input_size=hidden_a, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
                if 'v' in self.modals:
                    hidden_v = D_g
                    self.linear_v = nn.Linear(D_m_v, hidden_v)
                    if self.av_using_lstm:
                        self.lstm_v = nn.LSTM(input_size=hidden_v, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
                if 'l' in self.modals:
                    hidden_l = D_g
                    self.linear_l = nn.Linear(D_m, hidden_l)
                    self.lstm_l = nn.LSTM(input_size=hidden_l, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)

        else:
            print ('Base model must be one of DialogRNN/LSTM/GRU')
            raise NotImplementedError
        if self.modality_self_attention:
            
            self.a_align = MultiHeadCrossModalAttention(D_g, D_g, D_g, 2) 
            self.v_align = MultiHeadCrossModalAttention(D_g, D_g, D_g, 2) 
            self.l_align = MultiHeadCrossModalAttention(D_g, D_g, D_g, 2) 
        
        else:
            self.align = MultiHeadCrossModalAttention(D_g, D_g, D_g, 2) 

        self.graph_model = GCN(n_dim=D_g, nhidden=graph_hidden_size, 
                                        dropout=self.dropout, lamda=0.5, alpha=0.1, variant=True, return_feature=self.return_feature, use_residue=self.use_residue, n_speakers=n_speakers, modals=self.modals, use_speaker=self.use_speaker, use_modal=self.use_modal, num_L=num_L, num_K=num_K, original_gcn=self.original_gcn, graph_masking=self.graph_masking)
        print("construct "+self.graph_type)

        if self.multi_modal:
            self.dropout_ = nn.Dropout(self.dropout)
            self.hidfc = nn.Linear(graph_hidden_size, n_classes)
            if self.att_type == 'concat_subsequently':
                if self.use_residue:
                    self.smax_fc = nn.Linear((D_g+graph_hidden_size)*len(self.modals), n_classes)
                else:
                    self.smax_fc = nn.Linear((graph_hidden_size)*len(self.modals), n_classes)
            elif self.att_type == 'concat_DHT':
                if self.use_residue:
                    self.smax_fc = nn.Linear((D_g+graph_hidden_size*2)*len(self.modals), n_classes)
                else:
                    self.smax_fc = nn.Linear((graph_hidden_size*2)*len(self.modals), n_classes)
            elif self.att_type == 'gated':
                if len(self.modals) == 3:
                    self.smax_fc = nn.Linear(100*len(self.modals), graph_hidden_size)
                else:
                    self.smax_fc = nn.Linear(100, graph_hidden_size)
            else:
                self.smax_fc = nn.Linear(D_g+graph_hidden_size*len(self.modals), graph_hidden_size)


    def _reverse_seq(self, X, mask):
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)


    def forward(self, U, qmask, umask, seq_lengths, U_a=None, U_v=None, epoch=None):

        #=============roberta features
        [r1,r2,r3,r4]=U
        seq_len, _, feature_dim = r1.size()

        r1 = self.normBNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r2 = self.normBNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r3 = self.normBNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r4 = self.normBNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)

        U = (r1 + r2 + r3 + r4)/4
        #U = torch.cat((textf,acouf),dim=-1)
        #=============roberta features
        # print(self.multi_modal)
        if self.base_model == 'LSTM':
            if not self.multi_modal:
                if self.modals == "l":
                    U = self.linear_l(U)
                    emotions, _ = self.lstm(U)
                elif self.modals == "v":
                    U_v = self.linear_v(U_v)
                    emotions = U_v
                elif self.modals == "a":
                    U_a = self.linear_a(U_a)
                    emotions = U_a
            else:
                if self.av_using_lstm:
                    if 'a' in self.modals:
                        
                        U_a = self.linear_a(U_a)
                        
                        emotions_a, _ = self.lstm_a(U_a)
                        
                    if 'v' in self.modals:
                        U_v = self.linear_v(U_v)
                        emotions_v, _ = self.lstm_v(U_v)
                else:
                    
                    emotions_a = self.linear_a(U_a)
                    
                    emotions_v = self.linear_v(U_v)

                if 'l' in self.modals:
                    
                    U = self.linear_l(U)
                    
                    emotions_l, _ = self.lstm_l(U)
                    
                    
                if self.modality_self_attention:
                    emotions_a = self.a_align(emotions_a, emotions_a)
                    emotions_v = self.v_align(emotions_v, emotions_v)
                    emotions_l = self.l_align(emotions_l, emotions_l)
                
                else:
                    emotions_a = self.align(emotions_a, emotions_l) 
                    emotions_v = self.align(emotions_v, emotions_l) 
                    
        if not self.multi_modal:
            
            features = simple_batch_graphify(emotions, seq_lengths, self.no_cuda)
        else:
            if 'a' in self.modals:
                features_a = simple_batch_graphify(emotions_a, seq_lengths, self.no_cuda)
                
            else:
                features_a = []
            if 'v' in self.modals:
                features_v = simple_batch_graphify(emotions_v, seq_lengths, self.no_cuda)
            else:
                features_v = []
            if 'l' in self.modals:
                
                
                features_l = simple_batch_graphify(emotions_l, seq_lengths, self.no_cuda)
                
                
            else:
                features_l = []
        if self.graph_type=='hyper':
            emotions_feat = self.graph_model(features_a, features_v, features_l, seq_lengths, qmask, epoch)
            
            emotions_feat = self.dropout_(emotions_feat)
            
            emotions_feat = nn.ReLU()(emotions_feat)
            
            
            # print(emotions_feat.shape)
            log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)
            
        else:
            print("There are no such kind of graph")   
            
        
        return log_prob


#그래프 기준으로 unimodal 모델부터
#text부터

class UNIMODALModel(nn.Module):

    def __init__(self, base_model, D_m, D_g, D_e, graph_hidden_size, n_speakers, 
                 modality = None, #text, audio, visual
                 early_fusion = True,
                 n_classes=7, 
                 dropout=0.5, 
                 avec=False, 
                 no_cuda=False, 
                 graph_type='relation', 
                 use_topic=False,
                 alpha=0.2, 
                 multiheads=6, 
                 graph_construct='direct', 
                 use_GCN=False,
                 use_residue=True,
                 dynamic_edge_w=False,
                 D_m_v=512,
                 D_m_a=100,
                #  modals='avl',
                 att_type='gated',
                 av_using_lstm=False, 
                 dataset='IEMOCAP',
                 use_speaker=True, 
                 use_modal=False, 
                 num_L = 3, 
                 num_K = 4):
        
        super(UNIMODALModel, self).__init__()
        self.modality = modality
        self.early_fusion = early_fusion
        self.base_model = base_model
        self.avec = avec
        self.no_cuda = no_cuda
        self.graph_type=graph_type
        self.alpha = alpha
        self.multiheads = multiheads
        self.graph_construct = graph_construct
        self.use_topic = use_topic
        self.dropout = dropout
        self.use_GCN = use_GCN
        self.use_residue = use_residue
        self.dynamic_edge_w = dynamic_edge_w
        self.return_feature = True
        # self.modals = [x for x in modals]  # a, v, l
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.att_type = att_type
        
        if self.modality=="text":
            self.normBNa = nn.BatchNorm1d(1024, affine=True)
            self.normBNb = nn.BatchNorm1d(1024, affine=True)
            self.normBNc = nn.BatchNorm1d(1024, affine=True)
            self.normBNd = nn.BatchNorm1d(1024, affine=True)

        # if self.att_type == 'gated' or self.att_type == 'concat_subsequently' or self.att_type == 'concat_DHT':
        #     self.multi_modal = True
        #     self.av_using_lstm = av_using_lstm
        # else:
        #     self.multi_modal = False
        self.dataset = dataset

        if self.base_model == 'LSTM':
            # if not self.multi_modal:
            #     if len(self.modals) == 3:
            #         hidden_ = 250
            #     elif ''.join(self.modals) == 'al':
            #         hidden_ = 150
            #     elif ''.join(self.modals) == 'vl':
            #         hidden_ = 150
            #     else:
            #         hidden_ = 100
            #     self.linear_ = nn.Linear(D_m, hidden_)
            #     self.lstm = nn.LSTM(input_size=hidden_, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            # else:
            if self.modality =="audio":
                hidden_a = D_g
                self.linear_a = nn.Linear(D_m_a, hidden_a)
                # if self.av_using_lstm:
                self.lstm_a = nn.LSTM(input_size=hidden_a, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
            if self.modality =="visual":
                hidden_v = D_g
                self.linear_v = nn.Linear(D_m_v, hidden_v)
                # if self.av_using_lstm:
                self.lstm_v = nn.LSTM(input_size=hidden_v, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
            if self.modality =="text":
                hidden_l = D_g
                self.linear_l = nn.Linear(D_m, hidden_l)
                self.lstm_l = nn.LSTM(input_size=hidden_l, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)

        else:
            print ('Base model must be one of DialogRNN/LSTM/GRU')
            raise NotImplementedError

        self.align = MultiHeadCrossModalAttention(D_g, D_g, D_g, 2) 

        self.graph_model = UNIMODALGCN(n_dim=D_g, nhidden=graph_hidden_size, 
                                        dropout=self.dropout, lamda=0.5, alpha=0.1,  return_feature=self.return_feature, use_residue=self.use_residue, n_speakers=n_speakers, use_speaker=self.use_speaker, use_modal=self.use_modal, num_L=num_L, num_K=num_K)
        print("construct "+self.graph_type)

        # if self.multi_modal:
        self.dropout_ = nn.Dropout(self.dropout)
        self.hidfc = nn.Linear(graph_hidden_size, n_classes)
        if self.att_type == 'concat_subsequently':
            if self.use_residue:
                self.smax_fc = nn.Linear((D_g+graph_hidden_size), n_classes)
            else:
                self.smax_fc = nn.Linear((graph_hidden_size), n_classes)
        elif self.att_type == 'concat_DHT':
            if self.use_residue:
                self.smax_fc = nn.Linear((D_g+graph_hidden_size*2), n_classes)
            else:
                self.smax_fc = nn.Linear((graph_hidden_size*2), n_classes)
        elif self.att_type == 'gated':
            if len(self.modals) == 3:
                self.smax_fc = nn.Linear(100, graph_hidden_size)
            else:
                self.smax_fc = nn.Linear(100, graph_hidden_size)
        else:
            self.smax_fc = nn.Linear(D_g+graph_hidden_size, graph_hidden_size)


    def _reverse_seq(self, X, mask):
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)


    def forward(self, U, qmask, umask, seq_lengths, epoch=None):

        #=============roberta features
        if self.modality == "text":
            [r1,r2,r3,r4]=U
            seq_len, _, feature_dim = r1.size()

            r1 = self.normBNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r2 = self.normBNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r3 = self.normBNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r4 = self.normBNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)

            U = (r1 + r2 + r3 + r4)/4
        #U = torch.cat((textf,acouf),dim=-1)
        #=============roberta features
        if self.base_model == 'LSTM':
            
            if self.modality == "audio":
                U = self.linear_a(U)
                emotions = U
                emotions, _ = self.lstm_a(emotions)
                # print(U.size())
                
            elif  self.modality=="visual":
                U = self.linear_v(U)
                emotions = U
                emotions, _ = self.lstm_v(emotions)
                # print(emotions.shape)
                
            elif self.modality=="text":
                U = self.linear_l(U)
                emotions = U
                emotions, _ = self.lstm_l(emotions)

        emotions = self.align(emotions, emotions)
        # print(emotions.shape, "kkk")
        
        # self.linear_l = nn.Linear(D_m, hidden_l)
        #         self.lstm_l
        # emotions_a = self.align(emotions_a, emotions_l) 
        # emotions_v = self.align(emotions_v, emotions_l) 
                    
        # if not self.multi_modal:
            
        #     features = simple_batch_graphify(emotions, seq_lengths, self.no_cuda)
        # else:
        
        features = simple_batch_graphify(emotions, seq_lengths, self.no_cuda)
            # if 'a' in self.modals:
            #     features_a = simple_batch_graphify(emotions_a, seq_lengths, self.no_cuda)
                
            # else:
            #     features_a = []
            # if 'v' in self.modals:
            #     features_v = simple_batch_graphify(emotions_v, seq_lengths, self.no_cuda)
            # else:
            #     features_v = []
            # if 'l' in self.modals:
            #     features_l = simple_batch_graphify(emotions_l, seq_lengths, self.no_cuda)
            # else:
            #     features_l = []
        if self.graph_type=='hyper':
            emotions_feat = self.graph_model(features, seq_lengths, qmask, epoch)
            
            emotions_feat = self.dropout_(emotions_feat)
            
            emotions_feat = nn.ReLU()(emotions_feat)
            # print(emotions_feat.size())
            # log_prob = self.smax_fc(emotions_feat)
            # print(log_prob.shape)
            # print(log_prob)
            log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)
            # print(log_prob.shape)
        else:
            print("There are no such kind of graph")   
            
        
        return log_prob
    
    
    
class ModelMKD(nn.Module):

    def __init__(self, base_model, D_m, D_g, D_e, graph_hidden_size, n_speakers,
                 n_classes=7, dropout=0.5, avec=False, 
                 no_cuda=False, graph_type='relation', use_topic=False, alpha=0.2, multiheads=6, graph_construct='direct', use_GCN=False,use_residue=True,
                 dynamic_edge_w=False,D_m_v=512,D_m_a=100,modals='avl',att_type='gated',av_using_lstm=False, dataset='IEMOCAP',
                 use_speaker=True, use_modal=False, num_L = 3, num_K = 4):
        
        super(ModelMKD, self).__init__()

        self.base_model = base_model
        self.avec = avec
        self.no_cuda = no_cuda
        self.graph_type=graph_type
        self.alpha = alpha
        self.multiheads = multiheads
        self.graph_construct = graph_construct
        self.use_topic = use_topic
        self.dropout = dropout
        self.use_GCN = use_GCN
        self.use_residue = use_residue
        self.dynamic_edge_w = dynamic_edge_w
        self.return_feature = True
        self.modals = [x for x in modals]  # a, v, l
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.att_type = att_type
        self.normBNa = nn.BatchNorm1d(1024, affine=True)
        self.normBNb = nn.BatchNorm1d(1024, affine=True)
        self.normBNc = nn.BatchNorm1d(1024, affine=True)
        self.normBNd = nn.BatchNorm1d(1024, affine=True)

        if self.att_type == 'gated' or self.att_type == 'concat_subsequently' or self.att_type == 'concat_DHT':
            self.multi_modal = True
            self.av_using_lstm = av_using_lstm
        else:
            self.multi_modal = False
        self.dataset = dataset

        if self.base_model == 'LSTM':
            if not self.multi_modal:
                if len(self.modals) == 3:
                    hidden_ = 250
                elif ''.join(self.modals) == 'al':
                    hidden_ = 150
                elif ''.join(self.modals) == 'vl':
                    hidden_ = 150
                else:
                    hidden_ = 100
                self.linear_ = nn.Linear(D_m, hidden_)
                self.lstm = nn.LSTM(input_size=hidden_, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            else:
                if 'a' in self.modals:
                    hidden_a = D_g
                    self.linear_a = nn.Linear(D_m_a, hidden_a)
                    if self.av_using_lstm:
                        self.lstm_a = nn.LSTM(input_size=hidden_a, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
                if 'v' in self.modals:
                    hidden_v = D_g
                    self.linear_v = nn.Linear(D_m_v, hidden_v)
                    if self.av_using_lstm:
                        self.lstm_v = nn.LSTM(input_size=hidden_v, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
                if 'l' in self.modals:
                    hidden_l = D_g
                    self.linear_l = nn.Linear(D_m, hidden_l)
                    self.lstm_l = nn.LSTM(input_size=hidden_l, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)

        else:
            print ('Base model must be one of DialogRNN/LSTM/GRU')
            raise NotImplementedError
        
        

        self.align = MultiHeadCrossModalAttention(D_g, D_g, D_g, 2) 

        self.graph_model = GCN(n_dim=D_g, nhidden=graph_hidden_size, 
                                        dropout=self.dropout, lamda=0.5, alpha=0.1, variant=True, return_feature=self.return_feature, use_residue=self.use_residue, n_speakers=n_speakers, modals=self.modals, use_speaker=self.use_speaker, use_modal=self.use_modal, num_L=num_L, num_K=num_K)
        print("construct "+self.graph_type)

        if self.multi_modal:
            self.dropout_ = nn.Dropout(self.dropout)
            self.hidfc = nn.Linear(graph_hidden_size, n_classes)
            if self.att_type == 'concat_subsequently':
                if self.use_residue:
                    self.smax_fc = nn.Linear((D_g+graph_hidden_size)*len(self.modals), n_classes)
                else:
                    self.smax_fc = nn.Linear((graph_hidden_size)*len(self.modals), n_classes)
            elif self.att_type == 'concat_DHT':
                if self.use_residue:
                    self.smax_fc = nn.Linear((D_g+graph_hidden_size*2)*len(self.modals), n_classes)
                else:
                    self.smax_fc_l = nn.Linear(graph_hidden_size*2, n_classes)
                    self.smax_fc_v = nn.Linear(graph_hidden_size*2, n_classes)
                    self.smax_fc_a = nn.Linear(graph_hidden_size*2, n_classes)
                    self.smax_fc = nn.Linear((graph_hidden_size*2)*len(self.modals), n_classes)
                    print(graph_hidden_size)
            elif self.att_type == 'gated':
                if len(self.modals) == 3:
                    self.smax_fc = nn.Linear(100*len(self.modals), graph_hidden_size)
                else:
                    self.smax_fc = nn.Linear(100, graph_hidden_size)
            else:
                self.smax_fc = nn.Linear(D_g+graph_hidden_size*len(self.modals), graph_hidden_size)


    def _reverse_seq(self, X, mask):
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)


    def forward(self, U, qmask, umask, seq_lengths, U_a=None, U_v=None, epoch=None):

        #=============roberta features
        [r1,r2,r3,r4]=U
        seq_len, _, feature_dim = r1.size()

        r1 = self.normBNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r2 = self.normBNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r3 = self.normBNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r4 = self.normBNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)

        U = (r1 + r2 + r3 + r4)/4
        #U = torch.cat((textf,acouf),dim=-1)
        #=============roberta features
        # print(self.multi_modal)
        if self.base_model == 'LSTM':
            if not self.multi_modal:
                if self.modals == "l":
                    U = self.linear_l(U)
                    emotions, _ = self.lstm(U)
                elif self.modals == "v":
                    U_v = self.linear_v(U_v)
                    emotions = U_v
                elif self.modals == "a":
                    U_a = self.linear_a(U_a)
                    emotions = U_a
            else:
                if self.av_using_lstm:
                    if 'a' in self.modals:
                        
                        U_a = self.linear_a(U_a)
                        
                        emotions_a, _ = self.lstm_a(U_a)
                        
                    if 'v' in self.modals:
                        U_v = self.linear_v(U_v)
                        emotions_v, _ = self.lstm_v(U_v)
                else:
                    
                    emotions_a = self.linear_a(U_a)
                    
                    emotions_v = self.linear_v(U_v)

                if 'l' in self.modals:
                    
                    U = self.linear_l(U)
                    
                    emotions_l, _ = self.lstm_l(U)
                    
                    

                emotions_a = self.align(emotions_a, emotions_l) 
                emotions_v = self.align(emotions_v, emotions_l) 
                    
        if not self.multi_modal:
            
            features = simple_batch_graphify(emotions, seq_lengths, self.no_cuda)
        else:
            if 'a' in self.modals:
                features_a = simple_batch_graphify(emotions_a, seq_lengths, self.no_cuda)
                
            else:
                features_a = []
            if 'v' in self.modals:
                features_v = simple_batch_graphify(emotions_v, seq_lengths, self.no_cuda)
            else:
                features_v = []
            if 'l' in self.modals:
                
                
                features_l = simple_batch_graphify(emotions_l, seq_lengths, self.no_cuda)
                
                
            else:
                features_l = []
        if self.graph_type=='hyper':
            emotions_feat = self.graph_model(features_a, features_v, features_l, seq_lengths, qmask, epoch)
            
            emotions_feat = self.dropout_(emotions_feat)
            
            emotions_feat = nn.ReLU()(emotions_feat)
            
            # print(emotions_feat.shape)
            # print(emotions_feat.shape)
            if self.training:
                length_emotions_feat = emotions_feat.shape[-1]
                uni_feat_length = length_emotions_feat//3
                emotions_feat_l = emotions_feat[:, 0:uni_feat_length]
                # print(emotions_feat.shape)
                # print(emotions_feat_l.shape)
                emotions_feat_a = emotions_feat[:, uni_feat_length:2*uni_feat_length]
                emotions_feat_v = emotions_feat[:, 2*uni_feat_length:]
                
                score_l = self.smax_fc_l(emotions_feat_l) 
                score_a = self.smax_fc_a(emotions_feat_a)
                score_v = self.smax_fc_v(emotions_feat_v)
                
                score_l = F.log_softmax(score_l,1)
                score_a = F.log_softmax(score_a, 1)
                score_v = F.log_softmax(score_v, 1)
                # print(score_l.sum())
                # print(score_a.sum())
                
                log_prob_l = score_l
                log_prob_a = score_a
                log_prob_v = score_v
            else:
                log_prob_l = log_prob_a = log_prob_v = None
            
            log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)
            # print(log_prob.shape)
            
        else:
            print("There are no such kind of graph")   
            
        
        return log_prob, log_prob_l, log_prob_a, log_prob_v
    
    
    
    
class ModelMRL(nn.Module):

    def __init__(self, base_model, D_m, D_g, D_e, graph_hidden_size, n_speakers,
                 n_classes=7, dropout=0.5, avec=False, 
                 no_cuda=False, graph_type='relation', use_topic=False, alpha=0.2, multiheads=6, graph_construct='direct', use_GCN=False,use_residue=True,
                 dynamic_edge_w=False,D_m_v=512,D_m_a=100,modals='avl',att_type='gated',av_using_lstm=False, dataset='IEMOCAP',
                 use_speaker=True, use_modal=False, num_L = 3, num_K = 4, MRL_efficient = True, MRL_random=True):
        
        super(ModelMRL, self).__init__()
        self.MRL_efficient = MRL_efficient
        
        self.MRL_random = MRL_random
        print("MRL random", MRL_random)
        print(self.MRL_efficient)
        self.base_model = base_model
        self.avec = avec
        self.no_cuda = no_cuda
        self.graph_type=graph_type
        self.alpha = alpha
        self.multiheads = multiheads
        self.graph_construct = graph_construct
        self.use_topic = use_topic
        self.dropout = dropout
        self.use_GCN = use_GCN
        self.use_residue = use_residue
        self.dynamic_edge_w = dynamic_edge_w
        self.return_feature = True
        self.modals = [x for x in modals]  # a, v, l
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.att_type = att_type
        self.normBNa = nn.BatchNorm1d(1024, affine=True)
        self.normBNb = nn.BatchNorm1d(1024, affine=True)
        self.normBNc = nn.BatchNorm1d(1024, affine=True)
        self.normBNd = nn.BatchNorm1d(1024, affine=True)

        if self.att_type == 'gated' or self.att_type == 'concat_subsequently' or self.att_type == 'concat_DHT':
            self.multi_modal = True
            self.av_using_lstm = av_using_lstm
        else:
            self.multi_modal = False
        self.dataset = dataset

        if self.base_model == 'LSTM':
            if not self.multi_modal:
                if len(self.modals) == 3:
                    hidden_ = 250
                elif ''.join(self.modals) == 'al':
                    hidden_ = 150
                elif ''.join(self.modals) == 'vl':
                    hidden_ = 150
                else:
                    hidden_ = 100
                self.linear_ = nn.Linear(D_m, hidden_)
                self.lstm = nn.LSTM(input_size=hidden_, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            else:
                if 'a' in self.modals:
                    hidden_a = D_g
                    self.linear_a = nn.Linear(D_m_a, hidden_a)
                    if self.av_using_lstm:
                        self.lstm_a = nn.LSTM(input_size=hidden_a, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
                if 'v' in self.modals:
                    hidden_v = D_g
                    self.linear_v = nn.Linear(D_m_v, hidden_v)
                    if self.av_using_lstm:
                        self.lstm_v = nn.LSTM(input_size=hidden_v, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
                if 'l' in self.modals:
                    hidden_l = D_g
                    self.linear_l = nn.Linear(D_m, hidden_l)
                    self.lstm_l = nn.LSTM(input_size=hidden_l, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)

        else:
            print ('Base model must be one of DialogRNN/LSTM/GRU')
            raise NotImplementedError

        self.align = MultiHeadCrossModalAttention(D_g, D_g, D_g, 2) 

        self.graph_model = GCN(n_dim=D_g, nhidden=graph_hidden_size, 
                                        dropout=self.dropout, lamda=0.5, alpha=0.1, variant=True, return_feature=self.return_feature, use_residue=self.use_residue, n_speakers=n_speakers, modals=self.modals, use_speaker=self.use_speaker, use_modal=self.use_modal, num_L=num_L, num_K=num_K)
        print("construct "+self.graph_type)

        if self.multi_modal:
            self.dropout_ = nn.Dropout(self.dropout)
            self.hidfc = nn.Linear(graph_hidden_size, n_classes)
            # if self.att_type == 'concat_subsequently':
            #     if self.use_residue:
            #         self.smax_fc = nn.Linear((D_g+graph_hidden_size)*len(self.modals), n_classes)
            #     else:
            #         self.smax_fc = nn.Linear((graph_hidden_size)*len(self.modals), n_classes)
            if self.att_type == 'concat_DHT':
                if self.use_residue:
                    self.smax_fc = nn.Linear((D_g+graph_hidden_size*2)*len(self.modals), n_classes)
                else:
                    self.smax_fc = nn.Linear((graph_hidden_size*2)*len(self.modals), n_classes)
            # elif self.att_type == 'gated':
            #     if len(self.modals) == 3:
            #         self.smax_fc = nn.Linear(100*len(self.modals), graph_hidden_size)
            #     else:
            #         self.smax_fc = nn.Linear(100, graph_hidden_size)
            # else:
            #     self.smax_fc = nn.Linear(D_g+graph_hidden_size*len(self.modals), graph_hidden_size)
        
        if self.MRL_efficient==True:
            hiddensizes = []
            temp_hiddensize = (graph_hidden_size*2)*len(self.modals) //2
            while temp_hiddensize >= 3:
                hiddensizes.append(temp_hiddensize)
                temp_hiddensize = temp_hiddensize // 2
            
            self.hiddensizes = hiddensizes
            
            self.last_dimension = (graph_hidden_size*2)*len(self.modals)
        
        else:
            hiddensizes = []
            temp_hiddensize = (graph_hidden_size*2)*len(self.modals) //2
            while temp_hiddensize >= 3:
                hiddensizes.append(temp_hiddensize)
                temp_hiddensize = temp_hiddensize // 2
            
            self.hiddensizes = hiddensizes
            self.last_layers = nn.ModuleList()
            for hiddensize in self.hiddensizes:
                self.last_layers.append(nn.Linear(hiddensize, n_classes))
                
            self.last_dimension = (graph_hidden_size*2)*len(self.modals)
        # print(self.hiddensizes)
        # print(self.last_dimension)
        # for i, layer in enumerate(self.last_layers):
        #     print(f"Layer {i}: in_features = {layer.in_features}, out_features = {layer.out_features}")

            


    def _reverse_seq(self, X, mask):
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)


    def forward(self, U, qmask, umask, seq_lengths, U_a=None, U_v=None, epoch=None, label=None, loss_function = None):

        #=============roberta features
        [r1,r2,r3,r4]=U
        seq_len, _, feature_dim = r1.size()

        r1 = self.normBNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r2 = self.normBNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r3 = self.normBNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r4 = self.normBNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)

        U = (r1 + r2 + r3 + r4)/4
        #U = torch.cat((textf,acouf),dim=-1)
        #=============roberta features
        # print(self.multi_modal)
        if self.base_model == 'LSTM':
            if not self.multi_modal:
                if self.modals == "l":
                    U = self.linear_l(U)
                    emotions, _ = self.lstm(U)
                elif self.modals == "v":
                    U_v = self.linear_v(U_v)
                    emotions = U_v
                elif self.modals == "a":
                    U_a = self.linear_a(U_a)
                    emotions = U_a
            else:
                if self.av_using_lstm:
                    if 'a' in self.modals:
                        
                        U_a = self.linear_a(U_a)
                        
                        emotions_a, _ = self.lstm_a(U_a)
                        
                    if 'v' in self.modals:
                        U_v = self.linear_v(U_v)
                        emotions_v, _ = self.lstm_v(U_v)
                else:
                    
                    emotions_a = self.linear_a(U_a)
                    
                    emotions_v = self.linear_v(U_v)

                if 'l' in self.modals:
                    
                    U = self.linear_l(U)
                    
                    emotions_l, _ = self.lstm_l(U)
                    
                    

                emotions_a = self.align(emotions_a, emotions_l) 
                emotions_v = self.align(emotions_v, emotions_l) 
                    
        if not self.multi_modal:
            
            features = simple_batch_graphify(emotions, seq_lengths, self.no_cuda)
        else:
            if 'a' in self.modals:
                features_a = simple_batch_graphify(emotions_a, seq_lengths, self.no_cuda)
                
            else:
                features_a = []
            if 'v' in self.modals:
                features_v = simple_batch_graphify(emotions_v, seq_lengths, self.no_cuda)
            else:
                features_v = []
            if 'l' in self.modals:
                
                
                features_l = simple_batch_graphify(emotions_l, seq_lengths, self.no_cuda)
                
                
            else:
                features_l = []
        if self.graph_type=='hyper':
            emotions_feat = self.graph_model(features_a, features_v, features_l, seq_lengths, qmask, epoch)
            
            emotions_feat = self.dropout_(emotions_feat)
            
            emotions_feat = nn.ReLU()(emotions_feat)
            
            if self.training:
                if self.MRL_efficient:
                    total_loss = 0.0
                    uni_modality_length = emotions_feat.shape[-1]//3
                    # print(uni_modality_length)
                    DIMENSION= emotions_feat.shape[-1]
                    if self.MRL_random==True:
                        # print("MRL_efficient", self.MRL_efficient, "MRL_random", self.MRL_random)
                        full_indices = torch.randperm(DIMENSION)
                        for k, hiddensize in enumerate(self.hiddensizes):
                            sub_index = full_indices[:hiddensize]
                            x_selected =emotions_feat[:, sub_index]
                            # print(sub_index.shape)
                            weight_selected =self.smax_fc.weight[:, sub_index]
                            
                            log_prob_selected = F.log_softmax(x_selected@weight_selected.T+self.smax_fc.bias,1)
                            
                            
                            total_loss += loss_function(log_prob_selected, label)
                    else:
                        for k, hiddensize in enumerate(self.hiddensizes):
                            # print("MRL_efficient", self.MRL_efficient, "MRL_random", self.MRL_random)
                            uni_hiddensize = hiddensize//3    
                            # print("여기여야 하는데")
                            # print(self.MRL_random)
                            x_selected = torch.cat([emotions_feat[:,i*uni_modality_length:i*uni_modality_length+uni_hiddensize] for i in range(3)], dim=-1)
                            
                            weight_selected = torch.cat([self.smax_fc.weight[:,i*uni_modality_length:i*uni_modality_length+uni_hiddensize] for i in range(3)], dim=1)
                            
                            log_prob_selected = F.log_softmax(x_selected@weight_selected.T+self.smax_fc.bias,1)
                            
                            
                            total_loss += loss_function(log_prob_selected, label)
                    
                    log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)
                    total_loss += loss_function(log_prob, label)
                    
                    return total_loss, log_prob
        
                else:
                    total_loss = 0.0
                    uni_modality_length = emotions_feat.shape[-1]//3
                    
                    for k, hiddensize in enumerate(self.hiddensizes):
                        uni_hiddensize = hiddensize//3    
                        
                        x_selected = torch.cat([emotions_feat[:,i*uni_modality_length:i*uni_modality_length+uni_hiddensize] for i in range(3)], dim=-1)
                        
                        log_prob_selected = F.log_softmax(self.last_layers[k](x_selected), 1)
                        total_loss += loss_function(log_prob_selected, label)
                    
                    log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)
                    total_loss += loss_function(log_prob, label)
                    
                    return total_loss, log_prob
            else:
                log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)
                # print(log_prob.shape)
                # print(len(label))
                return log_prob
        else:
            print("There are no such kind of graph")   
            
        
        # return log_prob
