#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# In[5]:


class PACDataset(Dataset):
    def __init__(self, mode, data_path = "../data/pacs", reload=False, cache_folder_name='cache', split='train', train_test_split=0.7, seed=0):
        super(PACDataset).__init__()
        self.split = split
        if mode == 'p':
            self.data_path = os.path.join(os.path.abspath(data_path), 'photo')
            self.save_dat_name = 'photo'
        elif mode == 'a':
            self.data_path = os.path.join(os.path.abspath(data_path), 'art_painting')
            self.save_dat_name = 'art_painting'
        elif mode == 'c':
            self.data_path = os.path.join(os.path.abspath(data_path), 'cartoon')
            self.save_dat_name = 'cartoon'
        elif mode == 's':
            self.data_path = os.path.join(os.path.abspath(data_path), 'sketch')
            self.save_dat_name = 'sketch'
        
        
        self.cache_folder_name = cache_folder_name
        cache_folder = os.path.join(os.path.abspath(data_path), cache_folder_name)
        os.makedirs(cache_folder, exist_ok=True)
        cached_file_name_x_train = "cached_{}_x_train.npy".format(self.save_dat_name)
        cached_file_name_y_train = "cached_{}_y_train.npy".format(self.save_dat_name)
        cached_file_name_x_test = "cached_{}_x_test.npy".format(self.save_dat_name)
        cached_file_name_y_test = "cached_{}_y_test.npy".format(self.save_dat_name)
        cached_file_x_path_train = os.path.join(cache_folder, cached_file_name_x_train)
        cached_file_y_path_train = os.path.join(cache_folder, cached_file_name_y_train)
        cached_file_x_path_test = os.path.join(cache_folder, cached_file_name_x_test)
        cached_file_y_path_test = os.path.join(cache_folder, cached_file_name_y_test)
        is_x_file_exist_train = os.path.exists(cached_file_x_path_train)
        is_y_file_exist_train = os.path.exists(cached_file_x_path_train)   
        is_x_file_exist_test = os.path.exists(cached_file_x_path_test)
        is_y_file_exist_test = os.path.exists(cached_file_x_path_test) 
        is_file_missing = not all([is_x_file_exist_train, is_y_file_exist_train, is_x_file_exist_test, is_y_file_exist_test])
        if is_file_missing or reload:
            self.reload = True
        else:
            self.reload = False
        
        category_list = [i for i in os.listdir(self.data_path) if i != self.cache_folder_name]
        self.category_dict = dict(enumerate(category_list))
        
        if self.reload:
            all_x =[]
            all_y = []
            

            for index, category in enumerate(category_list):
                print("Loading {}/{}...".format(self.save_dat_name, category))
                category_path = os.path.join(self.data_path, category)
                figure_list = os.listdir(category_path)
                all_x_cate = []
                for filename in tqdm(figure_list):
                    figure_path = os.path.join(category_path, filename)
                    figure = Image.open(figure_path)
                    x = TF.to_tensor(figure).unsqueeze(0)
                    #print(x.shape)
                    all_x_cate.append(x)
                # print("each catetory: {}".format(torch.cat(all_x_cate, axis=0).shape))
                all_x.append(torch.cat(all_x_cate, axis=0)) # (N, 3, 227, 227)
                all_y.append(torch.ones(len(all_x_cate)).long() * index) # (N, )
            x_all = torch.cat(all_x, axis=0)
            y_all = torch.cat(all_y, axis=0)
            self.len_all = y_all.shape[0]
            torch.manual_seed(seed)
            index_final = torch.randperm(self.len_all)
            print("index",index_final)
            x_all = x_all[index_final]
            y_all = y_all[index_final]
            self.x_train = x_all[:int(train_test_split * self.len_all)]
            self.y_train = y_all[:int(train_test_split * self.len_all)]
            self.x_test = x_all[int(train_test_split * self.len_all):]
            self.y_test = y_all[int(train_test_split * self.len_all):]
            
            print("Writing loaded dataset x_train {} to {}".format(self.x_train.shape, cached_file_x_path_train))
            np.save(cached_file_x_path_train, self.x_train.numpy())
            print("Writing loaded dataset y_train {} to {}".format(self.y_train.shape, cached_file_y_path_train))
            np.save(cached_file_y_path_train, self.y_train.numpy())
            print("Writing loaded dataset x_test {} to {}".format(self.x_test.shape, cached_file_x_path_test))
            np.save(cached_file_x_path_test, self.x_test.numpy())
            print("Writing loaded dataset y_test {} to {}".format(self.y_test.shape, cached_file_y_path_test))
            np.save(cached_file_y_path_test, self.y_test.numpy())
            
            if self.split == 'train':
                self.x = self.x_train
                self.y = self.y_train
                self.len = self.y_train.shape[0]
            elif self.split == 'test':
                self.x = self.x_test
                self.y = self.y_test
                self.len = self.y_test.shape[0]
                
            
        
        else:
            print("Loading {} {} data from {}".format(self.save_dat_name, self.split, cache_folder))
            if self.split == 'train':
                self.x = torch.tensor(np.load(cached_file_x_path_train))
                self.y = torch.tensor(np.load(cached_file_y_path_train))
                self.len = self.x.shape[0]
            elif self.split == 'test':
                self.x = torch.tensor(np.load(cached_file_x_path_test))
                self.y = torch.tensor(np.load(cached_file_y_path_test))
                self.len = self.x.shape[0]


        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

            
            
        
        
        


# In[6]:


class ConcatDataset(Dataset):
    def __init__(self, *argv):
        """
        *argv = (p_dataset_train, a_data_set_train, ...)
        """
        super(ConcatDataset).__init__()
        self.x = torch.cat([i.x for i in [*argv]], axis=0)
        self.y = torch.cat([i.y for i in [*argv]], axis=0)
        self.len = self.y.shape[0]
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        return self.x[index], self.y[index]


# In[22]:


class ConcatDomainDataset(Dataset):
    def __init__(self, *argv):
        """
        *argv = (p_dataset_train, a_data_set_train, ...)
        """
        super(ConcatDomainDataset).__init__()
        self.x = torch.cat([i.x for i in [*argv]], axis=0)
        self.y = torch.cat([(torch.ones(j.y.shape[0]).long() * i) for i,j in enumerate([*argv])], axis=0)
        self.len = self.y.shape[0]
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        return self.x[index], self.y[index]


# In[11]:


class ShuffleClassDataset(Dataset):
    def __init__(self, concat_dataset, seed=10):
        super(ShuffleClassDataset).__init__()        
        self.concat = concat_dataset # instance of ConcatDataset
        self.x = self.concat.x
        self.origin_y = self.concat.y
        torch.manual_seed(seed)
        index_permute = torch.randperm(self.origin_y.shape[0])
        self.y = self.origin_y[index_permute]
        self.len = self.x.shape[0]
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
        
        


# In[8]:


if __name__ == "__main__":
    p_dataset_train = PACDataset('p', split='train')
    a_dataset_train = PACDataset('a', split='train')
    c_dataset_train = PACDataset('c', split='train')
    s_dataset_train = PACDataset('s', split='train')
    concat = ConcatDataset(p_dataset_train, a_dataset_train)
    shuffle_concat = ShuffleDataset(concat)


# In[ ]:




