import torch.nn as nn
from BasicModel import BasicModel

class GRU4Rec(BasicModel):

    def __init__(self, item_num, hidden_size, batch_size, seq_len=50, num_layers=3, dropout=0.1, args=None):
        super(GRU4Rec, self).__init__(item_num, hidden_size, batch_size, seq_len, beta=0, args=args)
        self.gru = nn.GRU(
                        input_size = self.hidden_size, 
                        hidden_size = self.hidden_size*2,
                        num_layers=num_layers, 
                        batch_first=True,
                        bias=False,
                    )
        self.dense = nn.Linear(hidden_size*2, hidden_size)
        self.emb_dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def forward(self, users, item_list, label_list, mask, train=True):

        item_eb = self.embeddings(item_list) # [b, s, h]
        item_seq_emb_dropout = self.emb_dropout(item_eb)

        output, fin_state = self.gru(item_seq_emb_dropout) # [b, s, h], [num_layers, b, h]
        item_len_list = mask.sum(dim=1)
        gru_output = self.dense(output)
        user_eb = self.gather_indexes(gru_output, item_len_list - 1)
        scores = self.calculate_score(user_eb)

        return user_eb, scores

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
