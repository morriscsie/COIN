import math
import sys
import time
import faiss
import numpy as np
import torch
import torch.nn as nn
import signal
from tqdm import tqdm
error_flag = {'sig':0}

def sig_handler(signum, frame):
    error_flag['sig'] = signum
    print("segfault core", signum)

signal.signal(signal.SIGSEGV, sig_handler)

from utils import get_DataLoader, get_exp_name, get_model, load_model, save_model, to_tensor

def evaluate(model, test_data, hidden_size, device, k=20, coef=None, item_cate_map=None, args=None):
    topN = k
    if coef is not None:
        coef = float(coef)
    
    gpu_indexs = [None]
    for i in range(1000):
        try:
            item_embs = model.output_items().cpu().detach().numpy()
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.device = device.index
           
            gpu_indexs[0] = faiss.GpuIndexFlatIP(res, hidden_size, flat_config)
            gpu_indexs[0].add(item_embs)
            if error_flag['sig'] == 0:
                break
            else:
                print("core received", error_flag['sig'])
                error_flag['sig'] = 0
        except Exception as e:
            print("error received", e)
        print("Faiss re-try", i)
        time.sleep(5)
    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    test_loop = tqdm(test_data ,position=0,leave=True,ncols=60,colour='green')
    for (users, targets, items, mask) in test_loop:
        user_embs, _ = model(users, to_tensor(items, device), None, to_tensor(mask, device), train=False) # shape=(batch_size, num_interest, embedding_dim)
        user_embs = user_embs.cpu().detach().numpy()
        gpu_index = gpu_indexs[0]
        if len(user_embs.shape) == 2:
            D, I = gpu_index.search(user_embs, topN)
            for i, iid_list in enumerate(targets): 
                recall = 0
                dcg = 0.0
                item_list = set(I[i]) 
                for no, iid in enumerate(item_list):
                    if iid in iid_list:
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
        else:
            ni = user_embs.shape[1]
            user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]]) # shape=(batch_size*num_interest, embedding_dim)
            D, I = gpu_index.search(user_embs, topN)
            for i, iid_list in enumerate(targets):
                recall = 0
                dcg = 0.0
                item_list_set = set()
                item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1)))
                item_list.sort(key=lambda x:x[1], reverse=True)
                for j in range(len(item_list)):
                    if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                        item_list_set.add(item_list[j][0])
                        if len(item_list_set) >= topN:
                            break                
                for no, iid in enumerate(item_list_set):
                    if iid in iid_list:
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
        total += len(targets)
  
    model.eval_flag = True
    recall = total_recall / total
    ndcg = total_ndcg / total
    hitrate = total_hitrate * 1.0 / total
    return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}


torch.set_printoptions(
    precision=2,
    threshold=np.inf,
    edgeitems=3,
    linewidth=200,
    profile=None,
    sci_mode=False
)

def train(device, train_file, valid_file, test_file, dataset, model_type, item_count, lr, seq_len, 
            hidden_size, interest_num, topN, max_iter, test_iter, decay_step, lr_decay, patience, exp, args):
    print("Param lr=" + str(lr))
    exp_name = get_exp_name(dataset, model_type, args.batch_size, lr, hidden_size, seq_len, interest_num, topN, exp=exp)
    if (args.save_path == None):
        best_model_path = "best_model/" + exp_name + '/'
    else:
        best_model_path = "best_model/" + args.save_path + '/'
    # prepare data
    train_data, train_user_nums = get_DataLoader(dataset, train_file, args.batch_size, seq_len, train_flag=1, start_idx=0)
    valid_data, valid_user_nums = get_DataLoader(dataset, valid_file, args.eval_batch_size, seq_len, train_flag=0, start_idx=train_user_nums)
    test_data, _ = get_DataLoader(dataset, test_file, args.eval_batch_size, seq_len, train_flag=0, start_idx=train_user_nums+valid_user_nums)
    model = get_model(dataset, model_type, item_count, args.batch_size, hidden_size, interest_num, seq_len, args=args, device=device)
    model = model.to(device)
    model.set_device(device)
    print("model beta = ", model.beta)
    model.set_sampler(args, device=device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=args.weight_decay)
    trials = 0
    print('training begin')
    sys.stdout.flush()
    start_time = time.time()
    model.loss_fct = loss_fn
    best_V_u = None
    if model_type == "COIN" and args.neighbors <= 0:
        raise ValueError("For model_type 'COIN', args.neighbors must be greater than 0.")
    if model_type != "COIN" and args.neighbors != 0:
        raise ValueError("For models other than 'COIN', args.neighbors must be 0.")
    if model_type == "COIN":
        build_neighbor = True
    else:
        build_neighbor = False
    try:
        total_loss = 0.0
        iter = 0
        best_metric = 0
        print(f"train_user_nums = {train_user_nums}")
        print(f"valid_user_nums = {valid_user_nums}")
        for epoch in range(args.epoch):
            train_loop = tqdm(train_data ,position=0, leave=True, ncols=60, colour='green', desc='Epoch '+ str(epoch + 1))
            model.train()
            iter = 0
            for (user_ids, targets, items, mask) in train_loop:
                iter += 1
                optimizer.zero_grad()
                pos_items = to_tensor(targets, device)
                interests, atten, readout, selection, contrastive_loss = None, None, None, None, None
                if model_type == "COIN":
                    interests, atten, readout, selection, contrastive_loss = model(user_ids, to_tensor(items, device), pos_items, to_tensor(mask, device))
                    base_loss = model.calculate_sampled_loss(readout, pos_items) 
                    attn_loss = model.calculate_atten_loss(atten, to_tensor(mask, device))
                    loss = base_loss + args.rlambda * attn_loss
                    if (contrastive_loss is not None):
                        loss += args.rcontrast * contrastive_loss
                if model_type in ["REMI", "REMI_rnn"]:
                    interests, atten, readout = model(user_ids, to_tensor(items, device), pos_items, to_tensor(mask, device))
                    loss = model.calculate_sampled_loss(readout, pos_items) 
                    loss += args.rlambda * model.calculate_atten_loss(atten)
                if model_type  == 'ComiRec-SA':
                    interests, readout, selection =  model(user_ids, to_tensor(items, device), pos_items, to_tensor(mask, device))
                    loss = model.calculate_sampled_loss(readout, pos_items) 
                if model_type == 'GRU4Rec':
                    readout, scores = model(user_ids, to_tensor(items, device), pos_items, to_tensor(mask, device))
                    loss = model.calculate_sampled_loss(readout, pos_items)
                loss.backward()
                optimizer.step()
                total_loss += loss
                if (build_neighbor):
                    for index, id in enumerate(user_ids):
                        model.V_u[id] = interests[index].detach() 

            if (build_neighbor):
                model.consider_neighbors = True
           
            print("Evaluation start")
            model.eval()
            metrics = evaluate(model, valid_data, hidden_size, device, topN, args=args)
            log_str = 'train loss: %.4f' % (total_loss / iter)
            if metrics != {}:
                log_str += ', ' + ', '.join(['valid ' + key + ': %.6f' % value for key, value in metrics.items()])
            print(log_str)
            if 'recall' in metrics:
                recall = metrics['recall']
                if recall > best_metric:
                    best_metric = recall
                    save_model(model, best_model_path)
                    trials = 0
                    import copy
                    if (build_neighbor):
                        best_V_u = copy.deepcopy(model.V_u)
                else:
                    trials += 1
                    if trials > patience:
                        print("early stopping!")
                        break
            total_loss = 0.0
            test_time = time.time()
            print("time interval: %.4f min" % ((test_time-start_time)/60.0))
            sys.stdout.flush()
    except KeyboardInterrupt:
        print('-' * 99)
        print('Exiting from training early')
    if (build_neighbor):
        torch.save(best_V_u, best_model_path + 'model_Vu.pth')
    load_model(model, best_model_path)
    if (build_neighbor):
        model.V_u = best_V_u
    model.eval()
    metrics = evaluate(model, valid_data, hidden_size, device, topN, args=args)
    print(', '.join(['Valid ' + key + ': %.6f' % value for key, value in metrics.items()]))
    print("Test result:")
    metrics = evaluate(model, test_data, hidden_size, device, 20, args=args)
    for key, value in metrics.items():
        print('test ' + key + '@20' + '=%.6f' % value)
    metrics = evaluate(model, test_data, hidden_size, device, 50, args=args)
    for key, value in metrics.items():
        print('test ' + key + '@50' + '=%.6f' % value)
    sys.stdout.flush()

def test(device, train_file, valid_file, test_file, cate_file, dataset, model_type, item_count, batch_size, lr, seq_len, 
            hidden_size, interest_num, topN, coef=None, exp='test', args=None):
    
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, hidden_size, seq_len, interest_num, topN, save=False, exp=exp)
    if (args.save_path == None):
        best_model_path = "best_model/" + exp_name + '/'
    else:
        best_model_path = "best_model/" + args.save_path + '/'
    model = get_model(dataset, model_type, item_count, batch_size, hidden_size, interest_num, seq_len, args=args)
    load_model(model, best_model_path)
    model = model.to(device)
    model.set_device(device)
    if (args.model_type == "COIN"):
        model.consider_neighbors = True
        model.V_u =  torch.load(best_model_path + 'model_Vu.pth', map_location='cpu')
    else:
        model.consider_neighbors = False
    model.eval()
    # prepare data
    train_data, train_user_nums = get_DataLoader(dataset, train_file, args.batch_size, seq_len, train_flag=1, start_idx=0)
    valid_data, valid_user_nums = get_DataLoader(dataset, valid_file, args.eval_batch_size, seq_len, train_flag=0, start_idx=train_user_nums)
    test_data, _ = get_DataLoader(dataset, test_file, args.eval_batch_size, seq_len, train_flag=0, start_idx=train_user_nums+valid_user_nums)
    metrics = evaluate(model, test_data, hidden_size, device, 20, args=args)
    for key, value in metrics.items():
        print('test ' + key + '@20' + '=%.6f' % value)
    metrics = evaluate(model, test_data, hidden_size, device, 50, args=args)
    for key, value in metrics.items():
        print('test ' + key + '@50' + '=%.6f' % value)
    sys.stdout.flush()