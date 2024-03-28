import pickle

import os
import re
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import warnings
from policy_eval.dataset import Dataset
import tensorflow as tf

class Mdp_Dataset(Dataset):
    def __init__(
            self,
            eval_length=48,
            target_dim=35,
            use_index_list=None,
            missing_ratio=0.0,
            seed=0,
            data_dir=None):
        self.eval_length = eval_length
        self.target_dim = target_dim
        np.random.seed(seed)  # seed for ground truth choice

        # target_cols = 1
        target_cols = target_dim + 1

        # if data_dir is None:
        #     # data_dir = "/home/caohaoqun/Conditional_Schrodinger_Bridge_Imputation/data/mydata.csv"
        #     # data_dir = '/home/caohaoqun/CSDI_replicant/mdp_mountaincar.csv'
        #     data_dir = '/home/caohaoqun/CSBI_replicant/training_datawalker2d-medium-v0.csv'

        # path = data_dir
        # print('Data path:', path)
        # try:
        #     df = pd.read_csv(data_dir, header=None).iloc[use_index_list,]
        # except:
        #     df = pd.read_csv(data_dir, header=None)


        # modification to benchmark/baseline
        data_dir = "/home/ec2-user/CSBI_replicant/policy_eval/trajectory_datasets"
        # env_name = 'Reacher-v2'
        env_name = 'Hopper-v2'
        behavior_policy_std = None
        num_trajectories = 1000
        normalize_states = True
        normalize_rewards = True
        noise_scale = 0.25
        bootstrap = True
        data_file_name = os.path.join(data_dir, env_name, '0',
                                  f'dualdice_{behavior_policy_std}.pckl')

        behavior_dataset = Dataset(
        data_file_name,
        num_trajectories,
        normalize_states=normalize_states,
        normalize_rewards=normalize_rewards,
        noise_scale=noise_scale,
        bootstrap=bootstrap)

        self.obs = self.meta_to_condtargetdata(behavior_dataset)
        ### done
        
        # df = df.iloc[:, 1:]
        # self.df = df.values
        # self.obs = df.values
        self.obs = self.obs.reshape(self.obs.shape[0],self.obs.shape[1], 1)
        self.cond_mask = np.zeros_like(self.obs)
        # self.cond_mask[:, : df.values.shape[1] - target_cols] = 1
        self.cond_mask[:, : self.obs.shape[1] - target_cols, 0] = 1
        self.gt_mask = np.zeros_like(self.cond_mask)


    def meta_to_condtargetdata(self,dataset):

        states = dataset.states
        next_states = dataset.next_states
        actions = dataset.actions
        rewards = dataset.rewards
        data = tf.concat([states, actions, next_states, tf.expand_dims(rewards, axis = 1)],axis = 1)
        return data.numpy()


    def __getitem__(self, org_index):
        
        s = {
            # "obs":self.obs[org_index],
            # "cond_mask":self.cond_mask[org_index]
            # ,"gt_mask":self.gt_mask[org_index]
            'observed_data':self.obs[org_index],
            'cond_mask':self.cond_mask[org_index]
            ,"gt_mask":self.gt_mask[org_index]
        }
        return s

    def __len__(self):
        return self.obs.shape[0]

 
def get_dataloader(
        seed=1,
        nfold=0,
        batch_size=16,
        eval_length=48,
        target_dim=36,
        missing_ratio=0.1,
        device='cpu',
        return_dataset=False):
    """Create dataloaders."""

    # only to obtain total length of dataset
    dataset = Mdp_Dataset(missing_ratio=missing_ratio, seed=seed)
    indlist = np.arange(len(dataset))

    np.random.seed(seed)
    np.random.shuffle(indlist)

    # 5-fold test
    start = (int)(nfold * 0.2 * len(dataset))
    end = (int)((nfold + 1) * 0.2 * len(dataset))
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.seed(seed)
    np.random.shuffle(remain_index)
    num_train = (int)(len(dataset) * 0.7)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    train_dataset = Mdp_Dataset(
        eval_length=eval_length, target_dim=target_dim, use_index_list=train_index,
        missing_ratio=missing_ratio, seed=seed)
    valid_dataset = Mdp_Dataset(
        eval_length=eval_length, target_dim=target_dim, use_index_list=valid_index,
        missing_ratio=missing_ratio, seed=seed)
    test_dataset = Mdp_Dataset(
        eval_length=eval_length, target_dim=target_dim, use_index_list=test_index,
        missing_ratio=missing_ratio, seed=seed)

    print(f'physio nfold{nfold}, missing_ratio{missing_ratio}')
    print('train/test/val num samples', len(train_dataset), len(valid_dataset), len(test_dataset))
    if return_dataset:
        return train_dataset, valid_dataset, test_dataset
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=1, num_workers=8)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0, num_workers=8)
        return train_loader, valid_loader, test_loader
