import argparse
import os
import random
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from GRU4Rec import GRU4Rec
from ComiRec import ComiRec_SA
from REMI import REMI
from COIN import COIN
def similarity(interest_emb, all_interest_emb):
    cos_vector = F.cosine_similarity(interest_emb, all_interest_emb, dim=2)
    values, idxs = torch.max(cos_vector, dim=1)
    return values, idxs
def neg_similarity(interest_emb, all_interest_emb):
    cos_vector = F.cosine_similarity(interest_emb, all_interest_emb, dim=2)
    values, idxs = torch.min(cos_vector, dim=1)
    return values, idxs
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=str, default='train', help='train | test') 
    parser.add_argument('--dataset', type=str, default='book', help='book | taobao') 
    parser.add_argument('--random_seed', type=int, default=2021)
    parser.add_argument('--hidden_size', type=int, default=64) 
    parser.add_argument('--interest_num', type=int, default=4)
    parser.add_argument('--model_type', type=str, default='REMI', help='DNN | GRU4Rec | MIND | ..') 
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning_rate') 
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--lr_dc_step', type=int, default=30, help='(k), the number of steps after which the learning rate decay')
    parser.add_argument('--max_iter', type=int, default=1000, help='(k)') 
    parser.add_argument('--patience', type=int, default=30) 
    parser.add_argument('--topN', type=int, default=50) 
    parser.add_argument('--gpu', type=str, default=None) 
    parser.add_argument('--coef', default=None)
    parser.add_argument('--exp', default='e1')
    parser.add_argument('--add_pos', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--sampled_n', type=int, default=1280)
    parser.add_argument('--sampled_loss', type=str, default='sampled')
    parser.add_argument('--sample_prob', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--hard_readout', type=int, default=1)
    parser.add_argument('--save_path', type=str, default=None, help='folder to save the model')
    # For REMI && COIN
    parser.add_argument('--rbeta', type=float, default=0)
    parser.add_argument('--rlambda', type=float, default=0)
    # For COIN
    parser.add_argument('--neighbors', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--rcontrast', type=float, default=0.2)
    parser.add_argument('--t_cont', type=float, default=1)
    parser.add_argument('--gru_layer', type=int, default=1)
    parser.add_argument('--gru_drop', type=float, default=0.0)
    parser.add_argument('--trm_layer', type=int, default=1)
    parser.add_argument('--trm_drop', type=float, default=0.0)
    parser.add_argument('--bias', action="store_true")
    return parser
class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_name, 
                 source,
                 seq_len=100,
                 train_flag=1,
                 start_idx=0
                ):
        self.read(source) 
        self.user_nums = len(self.users)
        self.train_flag = train_flag 
        self.seq_len = seq_len 
        self.start_idx = start_idx
        print("total user:", len(self.users))
        print("total items:", len(self.items))
        print("total train seqs:", len(self.train_seqs))
    def read(self, source):
        self.graph = {}
        self.time_graph = {}
        self.users = set()
        self.items = set()
        self.times = set()
        with open(source, 'r') as f:
            for line in f:
                conts = line.strip().split(',')
                try:
                    user_id = int(conts[0])
                    item_id = int(conts[1])
                    if len(conts) == 3:
                        time_stamp = int(conts[2])
                    else:
                        idx = int(conts[2])
                        time_stamp = int(conts[3])
                    self.users.add(user_id)
                    self.items.add(item_id)
                    self.times.add(time_stamp)
                    if user_id not in self.graph:
                        self.graph[user_id] = []
                    self.graph[user_id].append((item_id, time_stamp))
                except:
                    continue
        for user_id, value in self.graph.items(): 
            value.sort(key=lambda x: x[1])
            time_list = list(map(lambda x: x[1], value)) 
            time_min = min(time_list) 
            self.graph[user_id] = [x[0] for x in value] 
            self.time_graph[user_id] = [int(round((x[1] - time_min) / 86400.0) + 1) for x in value] 
        
        self.users = list(self.users) 
        self.items = list(self.items) 
        
        # Sliding window for training 
        self.train_seqs = []
        self.labels = []
        self.ids = []
        id = 0
        for user_id, items_seq in self.graph.items():
            x = min(len(items_seq) - 4, 10)
            label_positions = random.sample(range(4, len(items_seq)), x)
            for k in label_positions:
                self.labels += [items_seq[k]]
                self.train_seqs += [items_seq[:k]]
                self.ids += [id]
                id += 1
       
       
    def __len__(self):
        if (self.train_flag == 1):
            return len(self.train_seqs)
        else:
            return self.user_nums

    def __getitem__(self, idx):
        item_id_list = [] 
        hist_item_list = []
        hist_mask_list = []
        if (self.train_flag == 1):
            item_seqs = self.train_seqs[idx]
            k = len(item_seqs)
            item_id_list.append(self.labels[idx]) 
            user_id = self.ids[idx]
          
            if k >= self.seq_len: 
                hist_item_list.append(item_seqs[k-self.seq_len:])
                hist_mask_list.append([1.0] * self.seq_len)
            else:
                hist_item_list.append(item_seqs[:k] + [0] * (self.seq_len - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.seq_len - k))
         
            return user_id, item_id_list, hist_item_list, hist_mask_list 
        else:
            user_id = idx + self.start_idx 
            item_list = self.graph[user_id] 
            k = int(len(item_list) * 0.8)
            item_id_list.append(item_list[k:])
        
            if k >= self.seq_len: 
                hist_item_list.append(item_list[k-self.seq_len: k])
                hist_mask_list.append([1.0] * self.seq_len)
            else:
                hist_item_list.append(item_list[:k] + [0] * (self.seq_len - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.seq_len - k))
            return user_id, item_id_list, hist_item_list, hist_mask_list 

    def collate_fn(self, samples): 
        user_id_list = []
        item_id_list = []
        hist_item_list = []
        hist_mask_list = []

        for sample in samples:
            user_id_list.append(sample[0])
            item_id_list.append(sample[1][0])
            hist_item_list.append(sample[2][0])
            hist_mask_list.append(sample[3][0])
       
        return user_id_list, item_id_list, hist_item_list, hist_mask_list 


def get_DataLoader(dataset_name, source, batch_size, seq_len, train_flag, start_idx):
    dataset = Dataset(dataset_name, source, seq_len, train_flag, start_idx)
    if (train_flag == 1):
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=dataset.collate_fn), dataset.user_nums
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn), dataset.user_nums


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True



def get_model(dataset, model_type, item_count, batch_size, hidden_size, interest_num, seq_len, routing_times=3, args=None, device=None):

    add_pos = True 
    if args:
        add_pos = args.add_pos == 1
    if model_type == 'GRU4Rec':

        model = GRU4Rec(item_count, hidden_size, batch_size, seq_len, num_layers=args.layers, dropout=args.dropout, args=args)
    elif model_type in ['ComiRec-SA']:

        model = ComiRec_SA(item_count, hidden_size, batch_size, interest_num, seq_len, add_pos=add_pos, args = args, device = device)
    elif model_type == "REMI":
        model = REMI(item_count, hidden_size, batch_size, interest_num, seq_len, add_pos=add_pos, beta=args.rbeta,
                           args=args, device=device)
    elif model_type == "COIN":
        model = COIN(item_count, hidden_size, batch_size, interest_num, seq_len, add_pos=add_pos, beta=args.rbeta,
                           args=args, device=device)
    else:
        print ("Invalid model_type : %s", model_type)
        return
    model.name = model_type
    return model



def get_exp_name(dataset, model_type, batch_size, lr, hidden_size, seq_len, interest_num, topN, save=True, exp='e1'):
    extr_name = exp
    para_name = '_'.join([dataset, model_type, 'b'+str(batch_size), 'lr'+str(lr), 'd'+str(hidden_size), 
                            'len'+str(seq_len), 'in'+str(interest_num), 'top'+str(topN)])
    exp_name = para_name + '_' + extr_name

    while os.path.exists('best_model/' + exp_name) and save:
        shutil.rmtree('best_model/' + exp_name)
        break

    return exp_name


def save_model(model, Path):
    if not os.path.exists(Path):
        os.makedirs(Path)
    torch.save(model.state_dict(), Path + 'model.pt')


def load_model(model, path):
    model.load_state_dict(torch.load(path + 'model.pt'), strict=False)
    print('model loaded from %s' % path)


def to_tensor(var, device):
    var = torch.Tensor(var)
    var = var.to(device)
    return var.long()

def load_item_cate(source):
    item_cate = {}
    with open(source, 'r') as f:
        for line in f:
            conts = line.strip().split(',')
            item_id = int(conts[0])
            cate_id = int(conts[1])
            item_cate[item_id] = cate_id
    return item_cate


def compute_diversity(item_list, item_cate_map):
    n = len(item_list)
    diversity = 0.0
    for i in range(n):
        for j in range(i+1, n):
            diversity += item_cate_map[item_list[i]] != item_cate_map[item_list[j]]
    diversity /= ((n-1) * n / 2)
    return diversity
