# COIN:Collaborative Interest Modeling in Recommender Systems
Accepted at the 19th ACM Conference on Recommender Systems (**RecSys 2025**) 🎉📢

## 1. Introduction

This repository contains the implementation of our paper.

## 2. Acknowledgement

The structure of our code is based on [REMI](https://github.com/Tokkiu/REMI).

## 3. Dataset 
Original links of datasets:
- [Amazon](http://jmcauley.ucsd.edu/data/amazon/index.html)
  - Grocery 
- [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets)
  - [Gowalla](https://snap.stanford.edu/data/loc-gowalla.html)
  - [Tmall](https://tianchi.aliyun.com/dataset/53)

To guarantee fairness in the experiments, we strictly adhere to the dataset and preprocessing steps defined by the strongest baseline, REMI.

You can run `python process/data.py dataset_name` to preprocess the datasets.

## 4. Usage

### Environment Setup
```
$ conda env create -f env.yaml
```
### Data Preparation
Ensure that the `gowalla.inter` file is placed in the current working directory.  
You can download the dataset from: https://github.com/RUCAIBox/RecSysDatasets/blob/master/conversion_tools/usage/Gowalla.md
```
$ python process/data.py gowalla10
```

### Train and evaluate
Run the baseline.\
Available baselines:
- **REMI**
- ComiRec
- GRU4Rec

```
$ python3 src/train.py --model_type REMI -p train --gpu 0 --dataset gowalla10 --rlambda 100 --rbeta 1 --batch_size 256 \
--patience 10 --topN 20 --interest_num 4 --save_path baseline_gowalla_20 

$ python3 src/train.py --model_type REMI -p train --gpu 0 --dataset Grocery --rlambda 100 --rbeta 1 --batch_size 256 \
--patience 10 --topN 20 --interest_num 4 --save_path baseline_Grocery_20 

$ python3 src/train.py --model_type REMI -p train --gpu 0 --dataset tmall --rlambda 100 --rbeta 10 --batch_size 256 \
--patience 10 --topN 20 --interest_num 4 --save_path baseline_tmall_20 

```

Reproduce the reported result.
```
$ python3 src/train.py --model_type COIN -p train --gpu 0 --dataset gowalla10 --rlambda 100 --rbeta 1 --batch_size 256 --eval_batch_size 4 \
--alpha 0.5 --neighbors 4 --patience 10 --rcontrast 0.5 --t_cont 1 --topN 50 --save_path coin_gowalla_50 \
--interest_num 3 --trm_layer 1 --random_seed 999 --bias

$ python3 src/train.py --model_type COIN -p train --gpu 0 --dataset Grocery --rlambda 100 --rbeta 0.1 --batch_size 128 --eval_batch_size 4 \
--alpha 0.55 --neighbors 3 --patience 10 --rcontrast 10 --t_cont 0.1 --topN 50 --save_path coin_Grocery \
--interest 6 --trm_layer 1 --random_seed 2021

$ python3 src/train.py --model_type COIN -p train --gpu 0 --dataset tmall --rlambda 100 --rbeta 10 --batch_size 256 --eval_batch_size 1 \
--alpha 0.5 --neighbors 3 --patience 10 --rcontrast 0.5 --t_cont 1 --topN 50 --save_path coin_tmall_50 \
--interest 4 --trm_layer 1 --random_seed 2021
```
| Hyperparameter | Usage | 
| -------- | -------- | 
| interest | number of interest|
| rlambda  | routing regularization   |
| rbeta    | hard negative sampling|
| alpha    | coefficient of linear combination of neighbor interest|
| neighbors    | number of neighbor interest |
| patience    | early stopping  |
| rcontrast    | coefficient of contrastive learning loss|
| t_cont    | temperature of contrastive learning  |
| topN    | size of item candidate set  |
| trm_layer    | number of transformer layer  |

