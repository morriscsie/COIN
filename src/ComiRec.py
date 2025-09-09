import torch
import torch.nn as nn
import torch.nn.functional as F
from BasicModel import BasicModel
import utils
from functools import reduce

class ComiRec_SA(BasicModel):
    
    def __init__(self, item_num, hidden_size, batch_size, interest_num=4, seq_len=50, add_pos=True, beta=0, args=None, device=None):
        super(ComiRec_SA, self).__init__(item_num, hidden_size, batch_size, seq_len, beta, args)
        self.num_heads = interest_num
        self.interest_num = interest_num
        self.hard_readout = args.hard_readout
        self.add_pos = add_pos
        if self.add_pos:
            self.position_embedding = nn.Parameter(torch.Tensor(1, self.seq_len, self.hidden_size))
        self.linear1 = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 4, bias=False),
                nn.Tanh()
            )
        self.linear2 = nn.Linear(self.hidden_size * 4, self.num_heads, bias=False)
        self.reset_parameters()
        self.eval_flag = True
        self.all_interest = None
        self.neighbors = args.neighbors

    def forwardLogits(self, item_eb, mask):
        item_eb = item_eb * torch.reshape(mask, (-1, self.seq_len, 1))
        item_eb = torch.reshape(item_eb, (-1, self.seq_len, self.hidden_size))
        if self.add_pos:
            item_eb_add_pos = item_eb + self.position_embedding.repeat(item_eb.shape[0], 1, 1)
        else:
            item_eb_add_pos = item_eb

        # shape=(batch_size, maxlen, hidden_size*4)
        item_hidden = self.linear1(item_eb_add_pos)
        # shape=(batch_size, maxlen, num_heads)
        item_att_w = self.linear2(item_hidden)
        # shape=(batch_size, num_heads, maxlen)
        item_att_w = torch.transpose(item_att_w, 2, 1).contiguous()


        atten_mask = torch.unsqueeze(mask, dim=1).repeat(1, self.num_heads, 1)  # shape=(batch_size, num_heads, maxlen)
        paddings = torch.ones_like(atten_mask, dtype=torch.float) * (-2 ** 32 + 1)
        item_att_w = torch.where(torch.eq(atten_mask, 0), paddings, item_att_w)
        item_att_w = F.softmax(item_att_w, dim=-1)
        return item_att_w

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

        # shape=(batch_size, maxlen, hidden_size*4)
        item_hidden = self.linear1(item_eb_add_pos)
        # shape=(batch_size, maxlen, num_heads)
        item_att_w  = self.linear2(item_hidden)
        # shape=(batch_size, num_heads, maxlen)
        item_att_w  = torch.transpose(item_att_w, 2, 1).contiguous()

        atten_mask = torch.unsqueeze(mask, dim=1).repeat(1, self.num_heads, 1) # shape=(batch_size, num_heads, maxlen)
        paddings = torch.ones_like(atten_mask, dtype=torch.float) * (-2 ** 32 + 1)

        item_att_w = torch.where(torch.eq(atten_mask, 0), paddings, item_att_w)
        item_att_w = F.softmax(item_att_w, dim=-1)


        interest_emb = torch.matmul(item_att_w, # shape=(batch_size, num_heads, maxlen)
                                item_eb # shape=(batch_size, maxlen, embedding_dim)
                                ) # shape=(batch_size, num_heads, embedding_dim)

        user_eb = interest_emb # shape=(batch_size, num_heads, embedding_dim)

        if not train:
            if self.is_neighbor:
                if (self.eval_flag):
                    self.all_interest = None
                    all_interest_l = []
                    for _, Interest_matrix in self.V_u.items():
                        all_interest_l.append(torch.unsqueeze(Interest_matrix, 0))
                    self.all_interest = torch.cat(all_interest_l, dim=0)
                    self.eval_flag = False
                num_of_users = self.all_interest.size()[0]
                for i in range(user_eb.size()[0]):
                    user_matrix = user_eb[i]
                    user_matrixs = user_matrix.repeat((num_of_users, 1, 1))
                    scores = utils.similarity(user_matrixs, self.all_interest)
                    TopK = torch.topk(scores, self.neighbors)
                    readouts = []
                    weights = 0.0
                    for (idx, val) in (zip(TopK.indices, TopK.values)):
                        neighbor_emb = self.all_interest[idx]
                        readouts.append(val * neighbor_emb)
                        weights += val
                    neighbors_embs = reduce(torch.add, readouts)
                    neighbors_embs = torch.div(neighbors_embs, weights)
                    neighbors_embs = neighbors_embs * (1 - self.alpha)
                    user_eb[i] = user_matrix * self.alpha + neighbors_embs
                    del user_matrixs
                    del scores
                    torch.cuda.empty_cache()
                    
            return user_eb, None
      
        readout, selection, _ = self.read_out(users, user_eb, label_eb)
    
        return user_eb, readout, selection