#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.datasets as datasets
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader
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


# # Parser

# In[99]:


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
parser.add_argument('--model_path', type=str, help='where the data is stored')
parser.add_argument('--intervals', type=int, default=2, help='freq of compute f-div')
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--gfunction_epoch', type=int, default=5000, help='epoch of which gfunction is trained for')
parser.add_argument('--KL', type=bool, default=False, help="if calculate KL divergence")
parser.add_argument('--JS', type=bool, default=False, help="if calculate JS divergence")
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--classifier_epoch', type=int, default=10000, help='max iteration to train classifier')
parser.add_argument('--classifier', type=bool, default=True, help="if optmizer classifier")


args = parser.parse_args()
# snap shot of py file and command
python_file_name = sys.argv[0]


# # local only

# In[97]:


# # local only
# class local_args:
#     def __init__(self, **entries):
#         self.__dict__.update(entries)
        
# args = local_args(**{
#     'batch_size': 400,
#     'learning_rate': 1e-3,
#     'momentum': 0.5,
#     'gpu_num': 0,
#     'seed': 123,
#     'save_path': "../train_related",
#     'epochs': 2,
#     'subfolder': "test",
#     'wtarget': 0.7,
#     'dann_weight': 1,
#     'model_save_period': 2,
#     'start_shuffle_dann': 1,
#     'is_shuffle': 1,
#     'KL': True,
#     'model_path': '/home/tianqinl/Code/PGM-project/train_related/domain_shuffle_svhn_to_mnist/shuffle_weight_1.000000_learningrate_0.001000_startsepoch_30',
#     'classifier': True,
#     'intervals': 100,
#     'lr': 1e-3,
#     'JS': False,
#     'gfunction_epoch': 5,
#     'classifier_epoch': 100,
#     'sclass': 0.7
# })


# In[21]:



device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')
# seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
cudnn.deterministic = True
torch.backends.cudnn.deterministic = True



device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')
print(device)


model_sub_folder = args.subfolder + '/interval_%i'%(args.intervals)
save_folder = os.path.join(args.save_path, model_sub_folder)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)   


# In[22]:



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


# # Data loader

# In[7]:


mnist_trainset = datasets.MNIST(root='../data', train=True, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))


# In[8]:


mnist_testset = datasets.MNIST(root='../data', train=False, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))


# In[9]:


svhn_trainset = datasets.SVHN(root='../data', split='train', download=True, transform=torchvision.transforms.Compose([
                    torchvision.transforms.Resize((28, 28)),
                    torchvision.transforms.Grayscale(num_output_channels=1),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize([0.5], [0.5])]))


# In[10]:


svhn_testset = datasets.SVHN(root='../data', split='test', download=True, transform=torchvision.transforms.Compose([
                    torchvision.transforms.Resize((28, 28)),
                    torchvision.transforms.Grayscale(num_output_channels=1),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize([0.5], [0.5])]))


# In[23]:


# reload data
train_mnist_loader = DataLoader(mnist_trainset, batch_size=args.batch_size, shuffle=True)
test_mnist_loader = DataLoader(mnist_testset, batch_size=args.batch_size, shuffle=True)
train_svhn_loader = DataLoader(svhn_trainset, batch_size=args.batch_size, shuffle=True)
test_svhn_loader = DataLoader(svhn_testset, batch_size=args.batch_size, shuffle=True)


# ## Process data for cancat with source and target label

# In[24]:


class ConcatDataset(Dataset):
    def __init__(self, x, y, mode='mnist'):
        self.x = x
        self.y = y
        self.len = self.x.shape[0]
        self.mode = mode
        if self.mode == 'mnist':
            self.transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])
        elif self.mode == 'svhn':
            self.transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((28, 28)),
                    torchvision.transforms.Grayscale(num_output_channels=1),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        if self.mode == 'mnist':
            img = Image.fromarray(self.x[index].numpy(), mode='L')
            img = self.transform(img)
        elif self.mode == 'svhn':
            img = Image.fromarray(np.transpose(self.x[index], (1, 2, 0)))
            img = self.transform(img)
    
        return img, self.y[index]


# # Model

# In[25]:


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 30)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


# In[26]:


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

        


# In[27]:


class Adversial_loss(nn.Module):
    def __init__(self):
        super(Adversial_loss, self).__init__()
    
    def forward(self):
        pass


# In[28]:


def weights_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.LayerNorm:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


# In[91]:



device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')
print(device)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

encoder = Encoder().to(device)
CNet = FNN(d_in=30, d_h1=100, d_h2=100, d_out=10, dp=0.2).to(device)
DomainCNet = FNN(d_in=30, d_h1=1000, d_h2=100, d_out=2, dp=0.2).to(device)




optimizerEncoder = optim.Adam(encoder.parameters(), lr=args.learning_rate)
optimizerCNet = optim.Adam(CNet.parameters(), lr=args.learning_rate)
optimizerDomainCNet = optim.Adam(DomainCNet.parameters(), lr=args.learning_rate)

criterion_classifier = nn.CrossEntropyLoss().to(device)
# criterion_adverisal = 

encoder.apply(weights_init)
CNet.apply(weights_init)
DomainCNet.apply(weights_init)


# # Model Evaluation

# In[68]:


class Gfunction(nn.Sequential):
    def __init__(self):
        super(Gfunction, self).__init__(
            nn.Linear(30,100),
            nn.ELU(),
            nn.Linear(100,100),
            nn.ELU(),
            nn.Linear(100,100),
            nn.ELU(),
            nn.Linear(100,100),
            nn.ELU(),
            nn.Linear(100,100),
            nn.ELU(),
            nn.Linear(100,100),
            nn.ELU(),
            nn.Linear(100,100),
            nn.ELU(),
            nn.Linear(100,100),
            nn.ELU(),
            nn.Linear(100,100),
            nn.ELU(),
            nn.Linear(100,100),
            nn.ELU(),
            nn.Linear(100,100),
            nn.ELU(),
            nn.Linear(100,1)
        )


# In[69]:


def log_mean_exp(x, device):
    max_score = x.max()
    batch_size = torch.Tensor([x.shape[0]]).to(device)
    stable_x = x - max_score
    return max_score - batch_size.log() + stable_x.exp().sum(dim=0).log()

a = torch.rand([100,1]).to(device)
assert torch.all(log_mean_exp(a, device) - a.exp().mean(dim=0).log() < 1e-6)


# In[70]:


def KLDiv(g_x_source, g_x_target, device):
    # clipping
#     g_x_source = torch.clamp(g_x_source, -1e3, 1e3)
#     g_x_target = torch.clamp(g_x_target, -1e3, 1e3)
    return g_x_source.mean(dim=0) - log_mean_exp(g_x_target, device)


# In[71]:


if args.KL:
    gfunction_KL_div_labeled = Gfunction().to(device)
    gfunction_KL_div_unlabeled = Gfunction().to(device)


# # Estimate KL

# In[98]:


logger.info('Started loading')
total_epoch_trained = len([i for i in os.listdir(args.model_path) if i[0] == 'C'])


labeled_KL = []
unlabeled_KL = []
labeled_JS = []
unlabeled_JS = []
acc_source_unlabeled_classifier_ = []
acc_target_unlabeled_classifier_ = []

source_test_acc_ = []
target_test_acc_ = []
source_train_acc_ = []
epochs = []

for epoch in range(3, total_epoch_trained, args.intervals*args.model_save_period):
    epochs.append(epoch)
    # initialize 
    if args.KL:
        gfunction_KL_div_labeled.apply(weights_init)
        optimizer_gfunction_KL_div_labeled = torch.optim.Adam(gfunction_KL_div_labeled.parameters(), lr=args.lr)
        gfunction_KL_div_unlabeled.apply(weights_init)
        optimizer_gfunction_KL_div_unlabeled = torch.optim.Adam(gfunction_KL_div_unlabeled.parameters(), lr=args.lr)

    if args.JS:
        gfunction_JS_div_labeled.apply(weights_init)
        optimizer_gfunction_JS_div_labeled = torch.optim.Adam(gfunction_JS_div_labeled.parameters(), lr=args.lr)
        gfunction_JS_div_unlabeled.apply(weights_init)
        optimizer_gfunction_JS_div_unlabeled = torch.optim.Adam(gfunction_JS_div_unlabeled.parameters(), lr=args.lr)

    if args.classifier:
        CNet.load_state_dict(torch.load(os.path.join(args.model_path, 'CNet_%i.t7'%epoch)))
        optimizer_CNet = torch.optim.Adam(CNet.parameters(), lr=args.lr)
    
    # load weight
    encoder.load_state_dict(torch.load(os.path.join(args.model_path, 'encoder_%i.t7'%epoch)))
    
    # inferencing
    encoder.eval()
    
    # get source/target embedding
    source_x_labeled_embedding = torch.empty(0).to(device)
    source_y_labeled = torch.empty(0).long().to(device)
    source_x_unlabeled_embedding = torch.empty(0).to(device)
    source_y_unlabeled = torch.empty(0).long().to(device)
    target_x_labeled_embedding = torch.empty(0).to(device)
    target_y_labeled = torch.empty(0).long().to(device)
    target_x_unlabeled_embedding = torch.empty(0).to(device)
    target_y_unlabeled = torch.empty(0).long().to(device)
    with torch.no_grad():
        
        source_train_acc = 0.0
        num_data = 0.0
        for batch_id, (source_x, source_y) in tqdm(enumerate(train_svhn_loader), total=len(train_svhn_loader)):
            source_x = source_x.to(device).float()
            source_y = source_y.to(device).long()
            num_data += source_y.shape[0]
            source_x_embedding = encoder(source_x).detach()
            source_x_labeled_embedding = torch.cat([source_x_labeled_embedding, source_x_embedding])
            source_y_labeled = torch.cat([source_y_labeled, source_y])
            pred = CNet(source_x_embedding)
            source_train_acc += (pred.argmax(-1) == source_y).sum().item()
        source_train_acc = source_train_acc / num_data
        source_train_acc_.append(source_train_acc)
        
       
        
        source_test_acc = 0.0
        num_data = 0.0
        for batch_id, (source_x, source_y) in tqdm(enumerate(test_svhn_loader), total=len(test_svhn_loader)):
            source_x = source_x.to(device).float()
            source_y = source_y.to(device).long()
            num_data += source_y.shape[0]
            source_x_embedding = encoder(source_x).detach()
            source_x_unlabeled_embedding = torch.cat([source_x_unlabeled_embedding, source_x_embedding])
            source_y_unlabeled = torch.cat([source_y_unlabeled, source_y])
            pred = CNet(source_x_embedding)
            source_test_acc += (pred.argmax(-1) == source_y).sum().item()
        source_test_acc = source_test_acc / num_data
        source_test_acc_.append(source_test_acc)
       
        
        target_test_acc = 0.0
        num_data = 0.0
        for batch_id, (target_x, target_y) in tqdm(enumerate(test_mnist_loader), total=len(test_mnist_loader)):
            target_x = target_x.to(device).float()
            target_y = target_y.to(device).long()
            num_data += target_y.shape[0]
            fake_x_embedding = encoder(target_x).detach()
            target_x_unlabeled_embedding = torch.cat([target_x_unlabeled_embedding, fake_x_embedding])    
            target_y_unlabeled = torch.cat([target_y_unlabeled, target_y])
            pred = CNet(fake_x_embedding)
            target_test_acc += (pred.argmax(-1) == target_y).sum().item()
        target_test_acc = target_test_acc / num_data
        target_test_acc_.append(target_test_acc)
        
    # for loop to train the gfunction     
    for i in tqdm(range(args.gfunction_epoch)):
        if args.KL:
            optimizer_gfunction_KL_div_unlabeled.zero_grad()
            source_x_unlabeled_g = gfunction_KL_div_unlabeled(source_x_unlabeled_embedding)
            target_x_unlabeled_g = gfunction_KL_div_unlabeled(target_x_unlabeled_embedding)
            loss_KL_unlabeled = - KLDiv(source_x_unlabeled_g, target_x_unlabeled_g, device) # maximize
            loss_KL_unlabeled.backward()
            optimizer_gfunction_KL_div_unlabeled.step()
            
            if i % 500 == 0:
                print("Epoch %i, Iter %i, unlabeled KL: %f"%(epoch, i, -loss_KL_unlabeled.item()))

#         if args.JS:
#             optimizer_gfunction_JS_div_unlabeled.zero_grad()
#             source_x_unlabeled_g = gfunction_JS_div_unlabeled(source_x_unlabeled_embedding)
#             target_x_unlabeled_g = gfunction_JS_div_unlabeled(target_x_unlabeled_embedding)
#             loss_JS_unlabeled = - JSDiv(source_x_unlabeled_g, target_x_unlabeled_g, device) # maximize
#             loss_JS_unlabeled.backward()
#             optimizer_gfunction_JS_div_unlabeled.step()
#             if i % 500 == 0:
#                 print("Epoch %i, Iter %i, unlabeled JS: %f"%(epoch, i, -loss_JS_unlabeled.item()))
    if args.KL:  
        loss_KL_unlabeled = - loss_KL_unlabeled.item()
        unlabeled_KL.append(loss_KL_unlabeled)
    
#     if args.JS:
#         loss_JS_unlabeled = - loss_JS_unlabeled.item()
#         unlabeled_JS.append(loss_JS_unlabeled)
    
    
    
    acc_source_labeled_classifier = 0
    acc_target_labeled_classifier = 0
    if args.classifier:
#         while i < args.classifier_epoch or (acc_source_labeled_classifier < 0.98 and acc_target_labeled_classifier < 0.98):
#             i += 1

        for i in tqdm(range(args.classifier_epoch)):
            CNet.train()
            optimizer_CNet.zero_grad()
            pred = CNet(source_x_labeled_embedding)
            acc_source_labeled_classifier = (pred.argmax(-1) == source_y_labeled).sum().item() / pred.size(0)
            loss_source_classifier_labeled = criterion_classifier(pred, source_y_labeled)
            loss_source_classifier_labeled.backward()
            optimizer_CNet.step()
#             CNet.train()
#             encoder.train()
#             acc_source_labeled_classifier = 0.0            
#             num_datas = 0.0
#             for batch_id, (source_x, source_y) in tqdm(enumerate(train_svhn_loader), total=len(train_svhn_loader)):
#                 optimizerCNet.zero_grad()
#                 optimizerEncoder.zero_grad()
#                 source_x = source_x.to(device).float()
#                 source_y = source_y.to(device)
#                 num_datas += source_x.size(0)
#                 source_x_embedding = encoder(source_x)
#                 pred = CNet(source_x_embedding)
# #                 acc_source_labeled_classifier += (pred.argmax(-1) == source_y).sum().item()
#                 loss_source_classifier_labeled = criterion_classifier(pred, source_y)
#                 loss_source_classifier_labeled.backward()
#                 optimizerCNet.step()



#             if i % 500 == 0:
#                 CNet.eval()
#                 pred = CNet(source_x_unlabeled_embedding)
#                 acc_source_unlabeled_classifier = (pred.argmax(-1) == source_y_unlabeled).sum().item() / pred.size(0)
#                 pred = CNet(target_x_unlabeled_embedding)
#                 acc_target_unlabeled_classifier = (pred.argmax(-1) == target_y_unlabeled).sum().item() / pred.size(0)
#                 print("Iter %i: source acc: labeled: %f, unlabeled: %f; target acc: labeled: %f, unlabeled: %f"%(
#                     i, acc_source_labeled_classifier, acc_source_unlabeled_classifier, acc_target_labeled_classifier, acc_target_unlabeled_classifier))
        
        CNet.eval()
        pred = CNet(source_x_unlabeled_embedding)
        acc_source_unlabeled_classifier = (pred.argmax(-1) == source_y_unlabeled).sum().item() / pred.size(0)
        pred = CNet(target_x_unlabeled_embedding)
        acc_target_unlabeled_classifier = (pred.argmax(-1) == target_y_unlabeled).sum().item() / pred.size(0)
        acc_source_unlabeled_classifier_.append(acc_source_unlabeled_classifier)
        acc_target_unlabeled_classifier_.append(acc_target_unlabeled_classifier)
        
 
    
    logger.info("-----------------------------------------")
    log_string = "Epoch %i: "%epoch
    if args.KL: log_string += "testing KL: %f; "%(loss_KL_unlabeled)
    if args.JS: log_string += "testing JS: %f; "%(loss_JS_unlabeled)   
    if args.classifier: log_string += "src unlbl acc: %f, tgt unlbl acc: %f; "%(acc_source_unlabeled_classifier, acc_target_unlabeled_classifier)      
    logger.info(log_string)
    logger.info("-----------------------------------------")
    
    np.save(args.save_path+'/' +model_sub_folder+'/epochs.npy', epochs)
    np.save(args.save_path+'/' +model_sub_folder+'/source_test_acc_.npy', source_test_acc_)
    np.save(args.save_path+'/' +model_sub_folder+'/targhet_test_acc_.npy', target_test_acc_)
 
    if args.KL: 
        np.save(args.save_path+'/' +model_sub_folder+'/labeled_KL.npy', labeled_KL)
        np.save(args.save_path+'/' +model_sub_folder+'/unlabeled_KL.npy', unlabeled_KL)
        
    if args.JS:
        np.save(args.save_path+'/' +model_sub_folder+'/labeled_JS.npy', labeled_JS)
        np.save(args.save_path+'/' +model_sub_folder+'/unlabeled_JS.npy', unlabeled_JS)
        
    if args.classifier:
        np.save(args.save_path+'/' +model_sub_folder+'/acc_source_unlabeled_classifier_.npy', acc_source_unlabeled_classifier_)
        np.save(args.save_path+ '/' +model_sub_folder+'/acc_target_unlabeled_classifier_.npy', acc_target_unlabeled_classifier_)
    

