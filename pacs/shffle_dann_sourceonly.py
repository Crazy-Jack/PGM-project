#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.datasets as datasets
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader
from PAC_Dataset import PACDataset, ConcatDataset, ShuffleClassDataset, ConcatDomainDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import os
import sys
import logging
import logging.handlers
from PIL import Image
import torchvision.transforms.functional as TF
from shutil import copyfile


# # Parser

# In[2]:


parser = argparse.ArgumentParser(description='Domain adaptation')
parser.add_argument("--batch_size", type=int, default="400", help="batch size")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
parser.add_argument("--momentum", type=float, default=0.5, help="momentum")
parser.add_argument("--gpu_num", type=int, default=0, help="gpu num")
parser.add_argument("--seed", type=int, default=123, help="munually set seed")
parser.add_argument("--save_path", type=str, default="../train_related", help="save path")
parser.add_argument("--subfolder", type=str, default='test', help="subfolder name")
parser.add_argument("--wtarget", type=float, default=0.7, help="target weight")
parser.add_argument("--model_save_period", type=int, default=2, help="save period")
parser.add_argument("--epochs", type=int, default=2000, help="label shuffling")
parser.add_argument("--dann_weight", type=float, default=1, help="weight for label shuffling")
parser.add_argument("--start_shuffle_dann", type=int, default=100, help="when to start shuffling")
parser.add_argument("--is_shuffle", type=int, default=1, help="no shuffle if 0")
parser.add_argument("--domains", type=int, default=2, help="how many source domain")


args = parser.parse_args()
# snap shot of py file and command
python_file_name = sys.argv[0]


# # local only

# In[19]:


# # local only
# class local_args:
#     def __init__(self, **entries):
#         self.__dict__.update(entries)
        
# args = local_args(**{
#     'batch_size': 100,
#     'learning_rate': 1e-3,
#     'momentum': 0.5,
#     'gpu_num': 0,
#     'seed': 123,
#     'save_path': "../train_related",
#     'epochs': 20,
#     'subfolder': "test",
#     'wtarget': 0.7,
#     'dann_weight': 1,
#     'model_save_period': 2,
#     'start_shuffle_dann': 0,
#     'is_shuffle': 1,
#     'domains': 3,
# })


# In[5]:



device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')
# seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
cudnn.deterministic = True
torch.backends.cudnn.deterministic = True



device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')
print(device)


model_sub_folder = args.subfolder + '/shuffle_weight_%f_learningrate_%f_startsepoch_%i_isshuffle_%i_domains_%i'%(args.dann_weight, args.learning_rate, args.start_shuffle_dann, args.is_shuffle, args.domains)
save_folder = os.path.join(args.save_path, model_sub_folder)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)   


# In[6]:



logger = logging.getLogger()
logger.setLevel(logging.INFO)
logfile_path = os.path.join(save_folder, 'logfile.log')
if os.path.isfile(logfile_path):
    os.remove(logfile_path)
    
file_log_handler = logging.FileHandler(logfile_path)
logger.addHandler(file_log_handler)

stdout_log_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_log_handler)
logger.info("Fixed source testing bug")
attrs = vars(args)
for item in attrs.items():
    logger.info("%s: %s"%item)
logger.info("Training Save Path: {}".format(save_folder))

copyfile(python_file_name, os.path.join(save_folder, 'executed.py'))
commands = ['python']
commands.extend(sys.argv)
with open(os.path.join(save_folder, 'command.log'), 'w') as f:
    f.write(' '.join(commands))


# # Data loader

# In[7]:


p_dataset_train = PACDataset('p', split = 'train')
p_dataset_test = PACDataset('p', split = 'test')
a_dataset_train = PACDataset('a', split = 'train')
a_dataset_test = PACDataset('a', split = 'test')
c_dataset_train = PACDataset('c', split = 'train')
c_dataset_test = PACDataset('c', split = 'test')
s_dataset_train = PACDataset('s', split = 'train')
s_dataset_test = PACDataset('s', split = 'test')


# ## Visual comfirm

# In[8]:


# p_dataloader = DataLoader(p_dataset_test, batch_size=args.batch_size, shuffle=True)
# # pacs
# examples = enumerate(p_dataloader)
# batch_idx, (example_data, example_targets) = next(examples)


# fig = plt.figure()
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.tight_layout()
#     plt.imshow(example_data[i].permute(1, 2, 0))
#     plt.title("Ground Truth: {}-{}".format(example_targets[i], p_dataset_test.category_dict[int(example_targets[i])]))
#     plt.xticks([])
#     plt.yticks([])


# In[9]:


p_train_dataloader = DataLoader(p_dataset_train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
p_test_dataloader = DataLoader(p_dataset_test, batch_size=args.batch_size, shuffle=True, pin_memory=True)

a_train_dataloader = DataLoader(a_dataset_train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
a_test_dataloader = DataLoader(a_dataset_test, batch_size=args.batch_size, shuffle=True, pin_memory=True)

c_train_dataloader = DataLoader(c_dataset_train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
c_test_dataloader = DataLoader(c_dataset_test, batch_size=args.batch_size, shuffle=True, pin_memory=True)

s_train_dataloader = DataLoader(s_dataset_train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
s_test_dataloader = DataLoader(s_dataset_test, batch_size=args.batch_size, shuffle=True, pin_memory=True)


# In[ ]:





# # Shuffling Domain label

# In[10]:



source_train_tuple = (p_dataset_train, a_dataset_train, s_dataset_train)
# source_test_tuple = (p_dataset_test, a_dataset_test, s_dataset_test)

shuffle_domain_train_dataset = ShuffleClassDataset(ConcatDomainDataset(p_dataset_train, a_dataset_train, s_dataset_train, c_dataset_train))
# shuffle_domain_test = ShuffleClassDataset(ConcatDomainDataset(source_test_tuple))

shuffle_domain_train_dataloader = DataLoader(shuffle_domain_train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)


# # Model

# In[11]:


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=10, padding=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=11, padding=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(55*55*20, 1024)
        self.fc2 = nn.Linear(1024, 300)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 6, stride=2, padding=2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 6, stride=2, padding=2))
        x = x.view(-1, 55*55*20)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


# In[12]:


class FNN(nn.Module):
    def __init__(self, d_in, d_h1, d_h2, d_out, dp=0.2):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(d_in, d_h1)
        self.ln1 = nn.LayerNorm(d_h1)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dp)
        self.fc2 = nn.Linear(d_h1, d_h2)
        self.ln2 = nn.LayerNorm(d_h2)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dp)
        self.fc3 = nn.Linear(d_h2, d_out)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
    def before_lastlinear(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        return x

        


# In[13]:


def weights_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.LayerNorm:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


# In[14]:



device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')
print(device)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

encoder = Encoder().to(device)
CNet = FNN(d_in=300, d_h1=1000, d_h2=500, d_out=10, dp=0.2).to(device)
DomainCNet = FNN(d_in=300, d_h1=1000, d_h2=500, d_out=2, dp=0.2).to(device)




optimizerEncoder = optim.Adam(encoder.parameters(), lr=args.learning_rate)
optimizerCNet = optim.Adam(CNet.parameters(), lr=args.learning_rate)
optimizerDomainCNet = optim.Adam(DomainCNet.parameters(), lr=args.learning_rate)

criterion_classifier = nn.CrossEntropyLoss().to(device)
# criterion_adverisal = 

encoder.apply(weights_init)
CNet.apply(weights_init)
DomainCNet.apply(weights_init)


# # Train

# In[21]:


# acc store
source_acc_1_ = []
source_acc_2_ = []
source_acc_3_ = []
source_test_acc_1_ = []
source_test_acc_2_ = []
source_test_acc_3_ = []
target_test_acc_ = []
domain_acc_ = []

#loss store
accumulate_domain_loss_ = []
loss_source_1_ = []
loss_source_2_ = []
loss_source_3_ = []

logger.info('Started Training')


for epoch in range(args.epochs):
    log_loss_str = ""
    log_train_acc_str = ""
    log_test_acc_str = ""
    # update classifier
    # on source 1 domain p
    CNet.train()
    encoder.train()
    source_acc_1 = 0.0
    num_datas = 0.0
    loss_source_1 = 0.0
    for batch_id, (source_x, source_y) in tqdm(enumerate(p_train_dataloader), total=len(p_train_dataloader)):
        optimizerCNet.zero_grad()
        optimizerEncoder.zero_grad()
        source_x = source_x.to(device).float()
        source_y = source_y.to(device)
        num_datas += source_x.size(0)
        source_x_embedding = encoder(source_x)
        pred = CNet(source_x_embedding)
        source_acc_1 += (pred.argmax(-1) == source_y).sum().item()
        loss = criterion_classifier(pred, source_y)
        loss_source_1 += loss.item()
        loss.backward()
        optimizerCNet.step()
        optimizerEncoder.step()
        
        
    source_acc_1 = source_acc_1 / num_datas
    source_acc_1_.append(source_acc_1)
    loss_source_1_.append(loss_source_1)
    
    log_loss_str += "source 1 loss: {}; ".format(loss_source_1)
    log_train_acc_str += "source 1 train acc: {}; ".format(source_acc_1)
    
    if args.domains >= 2:
        # on source 2 domain c
        CNet.train()
        encoder.train()
        source_acc_2 = 0.0
        num_datas = 0.0
        loss_source_2 = 0.0
        for batch_id, (source_x, source_y) in tqdm(enumerate(a_train_dataloader), total=len(a_train_dataloader)):
            optimizerCNet.zero_grad()
            optimizerEncoder.zero_grad()
            source_x = source_x.to(device).float()
            source_y = source_y.to(device)
            num_datas += source_x.size(0)
            source_x_embedding = encoder(source_x)
            pred = CNet(source_x_embedding)
            source_acc_2 += (pred.argmax(-1) == source_y).sum().item()
            loss = criterion_classifier(pred, source_y)
            loss_source_2 += loss.item()
            loss.backward()
            optimizerCNet.step()
            optimizerEncoder.step()


        source_acc_2 = source_acc_2 / num_datas
        source_acc_2_.append(source_acc_2)
        loss_source_2_.append(loss_source_2)
        log_loss_str += "source 2 loss: {}; ".format(loss_source_2)
        log_train_acc_str += "source 2 train acc: {}; ".format(source_acc_2)
        
    if args.domains >= 3:
        # on source 3 domain s
        CNet.train()
        encoder.train()
        source_acc_3 = 0.0
        num_datas = 0.0
        loss_source_3 = 0.0
        for batch_id, (source_x, source_y) in tqdm(enumerate(s_train_dataloader), total=len(s_train_dataloader)):
            optimizerCNet.zero_grad()
            optimizerEncoder.zero_grad()
            source_x = source_x.to(device).float()
            source_y = source_y.to(device)
            num_datas += source_x.size(0)
            source_x_embedding = encoder(source_x)
            pred = CNet(source_x_embedding)
            source_acc_3 += (pred.argmax(-1) == source_y).sum().item()
            loss = criterion_classifier(pred, source_y)
            loss_source_3 += loss.item()
            loss.backward()
            optimizerCNet.step()
            optimizerEncoder.step()


        source_acc_3 = source_acc_3 / num_datas
        source_acc_3_.append(source_acc_3)
        loss_source_3_.append(loss_source_3)
        log_loss_str += "source 3 loss: {}; ".format(loss_source_3)
        log_train_acc_str += "source 3 train acc: {}; ".format(source_acc_3)
        
    # domain shuffle
    if args.is_shuffle != 0:
        accumulate_loss = 0.0
        domain_acc = 0.0
        DomainCNet.train()
        encoder.train()
        num_datas = 0.0
        for batch_id, (adv_x, adv_y) in tqdm(enumerate(shuffle_domain_train_dataloader), total=len(shuffle_domain_train_dataloader)):
            optimizerCNet.zero_grad()
            optimizerEncoder.zero_grad()
            adv_x = adv_x.to(device).float()
            adv_y = adv_y.to(device)
            num_datas += adv_x.size(0)
            adv_x_embedding = encoder(adv_x)
            pred = DomainCNet(adv_x_embedding)
            domain_acc += (pred.argmax(-1) == adv_y).sum().item()
            # adv_acc += (pred.argmax(-1) == adv_y).sum().item()
            loss = args.dann_weight * criterion_classifier(pred, adv_y)
            accumulate_loss += loss.item()
            loss.backward()
            optimizerDomainCNet.step()
            if epoch >= args.start_shuffle_dann:
                optimizerEncoder.step()    
        domain_acc = domain_acc / num_datas
        domain_acc_.append(domain_acc)
        log_train_acc_str += "shuffle domain acc: {}".format(domain_acc)
        accumulate_domain_loss_.append(accumulate_loss) 
        if epoch == args.start_shuffle_dann:
            logger.info("Start update Encoder using shuffling loss!")
        
        log_loss_str += "shuffle loss: {}; ".format(accumulate_loss)
    
    logger.info("Epoch {}: ".format(epoch) + log_loss_str)
    
          
        

    
    
    
    
    # eval on source   

    CNet.eval()
    encoder.eval()
    
    
    source_test_acc_1 = 0.0
    num_datas = 0.0    
    for batch_id, (source_x, source_y) in tqdm(enumerate(p_test_dataloader), total=len(p_test_dataloader)):
        optimizerCNet.zero_grad()
        optimizerEncoder.zero_grad()
        source_x = source_x.to(device).float()
        source_y = source_y.to(device)
        num_datas += source_x.size(0)
        source_x_embedding = encoder(source_x)
        pred = CNet(source_x_embedding)
        source_test_acc_1 += (pred.argmax(-1) == source_y).sum().item()
        
    source_test_acc_1 = source_test_acc_1 / num_datas
    source_test_acc_1_.append(source_test_acc_1)
    log_test_acc_str += "source 1 test acc: {}; ".format(source_test_acc_1)
    
    if args.domains >= 2:
        source_test_acc_2 = 0.0
        num_datas = 0.0    
        for batch_id, (source_x, source_y) in tqdm(enumerate(a_test_dataloader), total=len(a_test_dataloader)):
            optimizerCNet.zero_grad()
            optimizerEncoder.zero_grad()
            source_x = source_x.to(device).float()
            source_y = source_y.to(device)
            num_datas += source_x.size(0)
            source_x_embedding = encoder(source_x)
            pred = CNet(source_x_embedding)
            source_test_acc_2 += (pred.argmax(-1) == source_y).sum().item()

        source_test_acc_2 = source_test_acc_2 / num_datas
        source_test_acc_2_.append(source_test_acc_2)
        log_test_acc_str += "source 2 test acc: {}; ".format(source_test_acc_2)
    
    if args.domains >= 3:
        source_test_acc_3 = 0.0
        num_datas = 0.0    
        for batch_id, (source_x, source_y) in tqdm(enumerate(s_test_dataloader), total=len(s_test_dataloader)):
            optimizerCNet.zero_grad()
            optimizerEncoder.zero_grad()
            source_x = source_x.to(device).float()
            source_y = source_y.to(device)
            num_datas += source_x.size(0)
            source_x_embedding = encoder(source_x)
            pred = CNet(source_x_embedding)
            source_test_acc_3 += (pred.argmax(-1) == source_y).sum().item()

        source_test_acc_3 = source_test_acc_3 / num_datas
        source_test_acc_3_.append(source_test_acc_3)
        log_test_acc_str += "source 3 test acc: {}; ".format(source_test_acc_3)
    # eval on target 
    num_datas = 0.0
    target_test_acc = 0.0
    for batch_id, (target_x, target_y) in tqdm(enumerate(c_test_dataloader), total=len(c_test_dataloader)):
        optimizerCNet.zero_grad()
        optimizerEncoder.zero_grad()
        target_x = target_x.to(device).float()
        target_y = target_y.to(device)
        num_datas += target_x.size(0)
        target_x_embedding = encoder(target_x)
        pred = CNet(target_x_embedding)
        target_test_acc += (pred.argmax(-1) == target_y).sum().item()
    
    target_test_acc = target_test_acc / num_datas
    target_test_acc_.append(target_test_acc)
    log_test_acc_str += "target test acc: {}; ".format(target_test_acc)
    
    if epoch % args.model_save_period == 0:
        torch.save(DomainCNet.state_dict(), os.path.join(save_folder, 'DomainCNet_%i.t7'%(epoch+1)))
        torch.save(encoder.state_dict(), os.path.join(save_folder, 'encoder_%i.t7'%(epoch+1)))
        torch.save(CNet.state_dict(), os.path.join(save_folder, 'CNet_%i.t7'%(epoch+1)))


    logger.info("Epoch: {}; train ".format(epoch) + log_train_acc_str)
    logger.info("Epoch: {}; test ".format(epoch) + log_test_acc_str)
    
    np.save(os.path.join(args.save_path, model_sub_folder, 'source_acc_1_.npy'),source_acc_1_)
    np.save(os.path.join(args.save_path, model_sub_folder, 'source_acc_2_.npy'),source_acc_2_)
    np.save(os.path.join(args.save_path, model_sub_folder, 'source_acc_3_.npy'),source_acc_3_)
    
    np.save(os.path.join(args.save_path, model_sub_folder, 'source_test_acc_1_.npy'),source_test_acc_1_)
    np.save(os.path.join(args.save_path, model_sub_folder, 'source_test_acc_2_.npy'),source_test_acc_2_)
    np.save(os.path.join(args.save_path, model_sub_folder, 'source_test_acc_3_.npy'),source_test_acc_3_)
    
    np.save(os.path.join(args.save_path, model_sub_folder, 'target_test_acc_.npy'),target_test_acc_)
    np.save(os.path.join(args.save_path, model_sub_folder, 'domain_acc_.npy'),domain_acc_)
    np.save(os.path.join(args.save_path, model_sub_folder, 'accumulate_domain_loss_.npy'),accumulate_domain_loss_)
    
    np.save(os.path.join(args.save_path, model_sub_folder, 'loss_source_1_.npy'),loss_source_1_)
    np.save(os.path.join(args.save_path, model_sub_folder, 'loss_source_2_.npy'),loss_source_2_)
    np.save(os.path.join(args.save_path, model_sub_folder, 'loss_source_3_.npy'),loss_source_3_)


# In[ ]:




