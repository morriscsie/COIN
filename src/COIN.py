import torch
import torch.nn as nn
import torch.nn.functional as F
from BasicModel import BasicModel
import utils
from functools import reduce
class COIN(BasicModel):
    
    def __init__(self, item_num, hidden_size, batch_size, interest_num=4, seq_len=50, add_pos=True, beta=0, args=None, device=None):
        super(COIN, self).__init__(item_num, hidden_size, batch_size, seq_len, beta, args)
        self.num_heads = interest_num
        self.neighbors = args.neighbors
        self.gru_layer = args.gru_layer
        self.trm_layer = args.trm_layer
        self.hard_readout = args.hard_readout
        self.add_pos = add_pos
        self.alpha = args.alpha
        if self.add_pos:
            self.position_embedding = nn.Parameter(torch.Tensor(1, self.seq_len, self.hidden_size))
        self.linear1 = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size * 4, bias=args.bias),
                nn.Tanh()
        )
        self.linear2 = nn.Linear(self.hidden_size * 4, self.num_heads, bias=args.bias)
        self.reset_parameters()
        self.rnn = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, num_layers=self.gru_layer, bias=True, dropout=args.gru_drop, bidirectional=True)
        self._init_weights(self.rnn)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=1, dropout=args.trm_drop, batch_first=True)
        self.trm = nn.TransformerEncoder(encoder_layer, num_layers=self.trm_layer)
        self._init_weights(self.trm)
        self.eval_flag = True
        self.all_interest = None
        self.all_user_id = None
        

    def forward(self, users, item_list, label_list, mask, train=True):
        item_eb = self.embeddings(item_list)
        item_eb = item_eb * torch.reshape(mask, (-1, self.seq_len, 1))
        if train:
            label_eb = self.embeddings(label_list)

        item_eb = torch.reshape(item_eb, (-1, self.seq_len, self.hidden_size))

        if self.add_pos:
            item_eb_add_pos = item_eb + self.position_embedding.repeat(item_eb.shape[0], 1, 1)
        else:
            item_eb_add_pos = item_eb
        attn_mask = (mask == 0) 
        item_eb_add_pos = self.trm(item_eb_add_pos, src_key_padding_mask=attn_mask)  # shape = (batch_size, seq_len, hidden_size)
        item_eb_add_pos, _ = self.rnn(item_eb_add_pos)
       
        # shape=(batch_size, maxlen, hidden_size*4)
        item_hidden = self.linear1(item_eb_add_pos)
        # shape=(batch_size, maxlen, num_heads)
        item_att_w  = self.linear2(item_hidden)
        # shape=(batch_size, num_heads, maxlen)
        item_att_w  = torch.transpose(item_att_w, 2, 1).contiguous()
        
        atten_mask = torch.unsqueeze(mask, dim=1).repeat(1, self.num_heads, 1) # shape=(batch_size, num_heads, maxlen)
        paddings = torch.ones_like(atten_mask, dtype=torch.float) * (-2 ** 32 + 1)

        item_att_w_logits = torch.where(torch.eq(atten_mask, 0), paddings, item_att_w)
        item_att_w = F.softmax(item_att_w_logits, dim=-1)


        interest_emb = torch.matmul(item_att_w, # shape=(batch_size, num_heads, maxlen)
                                item_eb # shape=(batch_size, maxlen, embedding_dim)
                                ) # shape=(batch_size, num_heads, embedding_dim)


        user_eb = interest_emb # shape=(batch_size, num_heads, embedding_dim)

        if not train:
            if (self.eval_flag):
                self.all_interest = None
                all_interest_l = []
                all_user_id = []
                for user_id, interest_matrix in self.V_u.items():
                    all_interest_l.append(torch.unsqueeze(interest_matrix, 0).to(self.device)) 
                    all_user_id.append(user_id)
                self.all_interest = torch.cat(all_interest_l, dim=0)
                self.all_user_id = all_user_id
                self.eval_flag = False
            batch_size = user_eb.size(0)
            num_head = user_eb.size(1)
            list_of_neighbor_id = []
            list_of_neighbor_sim = []
            for i in range(batch_size):
                user_neighbor_ids = [] 
                user_neighbor_sim = []
                for j in range(num_head):
                    interest = user_eb[i][j]
                    chunks = torch.chunk(self.all_interest, 10, dim=0)
                    all_scores = torch.empty(0).to(self.device)
                    all_interest_idxs = torch.empty(0).long().to(self.device)
                    for chunk in chunks:
                        chunk_size = chunk.size(0)
                        interests = interest.unsqueeze(0).unsqueeze(0).repeat((chunk_size, self.num_heads, 1))
                        scores, interest_idxs = utils.similarity(interests, chunk)
                        all_scores = torch.cat((all_scores, scores), dim=0)
                        all_interest_idxs = torch.cat((all_interest_idxs, interest_idxs), dim=0)
                    TopK = torch.topk(all_scores, self.neighbors)                       
                    weights = 0.0
                    neighbors = []
                    neighbor_ids = [] 
                    neighbors_sim = []
                    for (idx, val) in zip(TopK.indices, TopK.values):
                        neighbor_id = self.all_user_id[idx]
                        neighbor_ids.append(neighbor_id)
                        interest_idx = all_interest_idxs[idx]
                        assert 0 <= interest_idx < num_head
                        neighbor_emb = self.all_interest[idx, interest_idx, :]
                        neighbors.append(val * neighbor_emb)
                        weights += val
                        neighbors_sim.append(val.item())
                    if (len(neighbors) > 0):
                        neighbors_embs = reduce(torch.add, neighbors)
                        neighbors_embs = torch.div(neighbors_embs, weights)
                        user_eb[i][j] = interest * 1 + neighbors_embs * self.alpha
                    else:
                        user_eb[i][j] = interest
                    user_neighbor_ids.append(neighbor_ids)
                    user_neighbor_sim.append(neighbors_sim)

                list_of_neighbor_id.append(user_neighbor_ids)
                list_of_neighbor_sim.append(user_neighbor_sim)
            return user_eb, list_of_neighbor_sim
        readout, selection, contrastive_loss = self.read_out(users, user_eb, label_eb)
        return user_eb, item_att_w, readout, selection, contrastive_loss

    def calculate_atten_loss(self, attention, item_mask):
        C_mean = torch.mean(attention, dim=2, keepdim=True)
        C_reg = (attention - C_mean)
        C_reg = torch.bmm(C_reg, C_reg.transpose(1, 2)) / self.hidden_size
        dr = torch.diagonal(C_reg, dim1=-2, dim2=-1)
        n2 = torch.norm(dr, dim=(1)) ** 2
        return n2.sum()