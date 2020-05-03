import torch
import torchvision
import torchvision.datasets as datasets
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader, RandomSampler
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
import random

# parser
parser = argparse.ArgumentParser(description='Domain adaptation')
parser.add_argument("--batch_size", type=int, default="100", help="batch size")
parser.add_argument("--learning_rate", type=float, default=1e-2, help="learning rate")
parser.add_argument("--momentum", type=float, default=0.5, help="momentum")
parser.add_argument("--gpu_num", type=int, default=0, help="gpu num")
parser.add_argument("--seed", type=int, default=123, help="munually set seed")
parser.add_argument("--save_path", type=str, default=r"C:\Users\Tianqin Li\Code\PGM-project\train_related", help="save path")
parser.add_argument("--subfolder", type=str, default=r'\baseline_st', help="subfolder name")
parser.add_argument("--wtarget", type=float, default=0.7, help="target weight")
parser.add_argument("--model_save_period", type=int, default=2, help="save period")

args = parser.parse_args()

# local only
class local_args:
    def __init__(self, **entries):
        self.__dict__.update(entries)

args = local_args(**{
    'batch_size': 400,
    'learning_rate': 1e-3,
    'momentum': 0.5,
    'gpu_num': 0,
    'seed': 123,
    'save_path': r"./",
    'epochs': 30,
    'subfolder': r'./',
    'wtarget': 0.7,
    'dann_weight': 1,
    'model_save_period': 2,
    'shuffle_weight': 0.1,
})

device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')
# seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
cudnn.deterministic = True
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')

model_sub_folder = args.subfolder + '\learningrate_%f_shuffle_weight_%f'%(args.learning_rate, args.shuffle_weight)

if not os.path.exists(args.save_path+model_sub_folder):
    os.makedirs(args.save_path+model_sub_folder)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if os.path.isfile(args.save_path+model_sub_folder+ '\logfile.log'):
    os.remove(args.save_path+model_sub_folder+ '\logfile.log')

file_log_handler = logging.FileHandler(args.save_path+model_sub_folder+ '\logfile.log')
logger.addHandler(file_log_handler)

stdout_log_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_log_handler)

attrs = vars(args)
for item in attrs.items():
    logger.info("%s: %s"%item)

# Download data
# mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.Compose([
#                                torchvision.transforms.ToTensor(),
#                                torchvision.transforms.Normalize(
#                                  (0.1307,), (0.3081,))
#                              ]))
# mnist_shuffle = datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.Compose([
#                                torchvision.transforms.ToTensor(),
#                                torchvision.transforms.Normalize(
#                                  (0.1307,), (0.3081,))
#                              ]))
# mnist_shuffle.targets = [random.randint(0, 9) for _ in range(len(mnist_shuffle))] # shuffle lable
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

svhn_trainset = datasets.SVHN(root='./data', split='train', download=True, transform=torchvision.transforms.Compose([
                    torchvision.transforms.Resize((28, 28)),
                    torchvision.transforms.Grayscale(num_output_channels=1),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize([0.5], [0.5])]))
svhn_shuffle = datasets.SVHN(root='./data', split='train', download=True, transform=torchvision.transforms.Compose([
                    torchvision.transforms.Resize((28, 28)),
                    torchvision.transforms.Grayscale(num_output_channels=1),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize([0.5], [0.5])]))
svhn_shuffle.targets = [random.randint(0, 9) for _ in range(len(svhn_shuffle))] # shuffle lable

svhn_testset = datasets.SVHN(root='./data', split='test', download=True, transform=torchvision.transforms.Compose([
                    torchvision.transforms.Resize((28, 28)),
                    torchvision.transforms.Grayscale(num_output_channels=1),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize([0.5], [0.5])]))


# load data
# my_sampler = RandomSampler(mnist_trainset) # passing the same sampler to ensure same shuffling
my_sampler = RandomSampler(svhn_trainset)

# train_mnist_loader = DataLoader(mnist_trainset, batch_size=args.batch_size, sampler = my_sampler)
# shuffle_mnist_loader = DataLoader(mnist_shuffle, batch_size=args.batch_size, sampler = my_sampler)
test_mnist_loader = DataLoader(mnist_testset, batch_size=args.batch_size)
train_svhn_loader = DataLoader(svhn_trainset, batch_size=args.batch_size, sampler = my_sampler)
shuffle_svhn_loader = DataLoader(svhn_shuffle, batch_size=args.batch_size, sampler = my_sampler)
test_svhn_loader = DataLoader(svhn_testset, batch_size=args.batch_size)

# encoder
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

# classifier
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

def weights_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.LayerNorm:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')
print(device)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

encoder1 = Encoder().to(device)
CNet1 = FNN(d_in=30, d_h1=100, d_h2=100, d_out=10, dp=0.2).to(device) # true classifier
encoder2 = Encoder().to(device)
CNet2 = FNN(d_in=30, d_h1=100, d_h2=100, d_out=10, dp=0.2).to(device) # true classifier

optimizerEncoder1 = optim.Adam(encoder1.parameters(), lr=args.learning_rate)
optimizerCNet1 = optim.Adam(CNet1.parameters(), lr=args.learning_rate)
optimizerEncoder2 = optim.Adam(encoder2.parameters(), lr=args.learning_rate)
optimizerCNet2 = optim.Adam(CNet2.parameters(), lr=args.learning_rate)
criterion_classifier = nn.CrossEntropyLoss().to(device)

encoder1.apply(weights_init)
CNet1.apply(weights_init)
encoder2.apply(weights_init)
CNet2.apply(weights_init)

# Train
train_acc1_ = []
train_acc2_ = []
svhn_test_acc1_ = []
svhn_test_acc2_ = []
mnist_test_acc1_ = []
mnist_test_acc2_ = []

logger.info('Started Training')

for epoch in range(args.epochs):
    train_acc1 = 0.0
    train_acc2 = 0.0
    num_datas = 0.0
    encoder1.train()
    CNet1.train()
    encoder2.train()
    CNet2.train()
    for batch_id, (source_x, source_y) in tqdm(enumerate(train_svhn_loader), total=len(train_svhn_loader)):
        # data
        source_x = source_x.to(device).float()
        source_y = source_y.to(device)
        num_datas += source_x.size(0)

        # clear gradient
        optimizerEncoder1.zero_grad()
        optimizerCNet1.zero_grad()

        # forward
        source_x_embedding = encoder1(source_x)
        pred1 = CNet1(source_x_embedding)
        train_acc1 += (pred1.argmax(-1) == source_y).sum().item()

        # backward
        loss1 = criterion_classifier(pred1, source_y)
        loss1.backward()
        optimizerCNet1.step()
        optimizerEncoder1.step()

    train_acc1 = train_acc1 / num_datas
    train_acc1_.append(train_acc1)

    for batch_id, (source_x, source_y) in tqdm(enumerate(shuffle_svhn_loader), total=len(shuffle_svhn_loader)):
        # data
        source_x = source_x.to(device).float()
        source_y = source_y.to(device)
        num_datas += source_x.size(0)

        # clear gradient
        optimizerEncoder2.zero_grad()
        optimizerCNet2.zero_grad()

        # forward
        source_x_embedding = encoder2(source_x)
        pred2 = CNet2(source_x_embedding)
        train_acc2 += (pred2.argmax(-1) == source_y).sum().item()

        # backward
        loss2 = criterion_classifier(pred2, source_y)
        loss2.backward()
        optimizerCNet2.step()
        optimizerEncoder2.step()

    train_acc2 = train_acc2/ num_datas
    train_acc2_.append(train_acc2)

    # eval on source
    source_test_acc1 = 0.0
    source_test_acc2 = 0.0
    num_datas = 0.0
    CNet1.eval()
    encoder1.eval()
    CNet2.eval()
    encoder2.eval()

    for batch_id, (source_x, source_y) in enumerate(test_svhn_loader):
        source_x = source_x.to(device).float()
        source_y = source_y.to(device)
        num_datas += source_x.size(0)
        source_x_embedding1 = encoder1(source_x)
        pred1 = CNet1(source_x_embedding1)
        source_test_acc1 += (pred1.argmax(-1) == source_y).sum().item()
        source_x_embedding2 = encoder2(source_x)
        pred2 = CNet2(source_x_embedding2)
        source_test_acc2 += (pred2.argmax(-1) == source_y).sum().item()

    source_test_acc1 = source_test_acc1 / num_datas
    svhn_test_acc1_.append(source_test_acc1)
    source_test_acc2 = source_test_acc2 / num_datas
    svhn_test_acc2_.append(source_test_acc2)

    # eval on source
    mnist_test_acc1 = 0.0
    mnist_test_acc2 = 0.0
    num_datas = 0.0
    for batch_id, (source_x, source_y) in enumerate(test_mnist_loader):
        source_x = source_x.to(device).float()
        source_y = source_y.to(device)
        num_datas += source_x.size(0)
        source_x_embedding1 = encoder1(source_x)
        pred1 = CNet1(source_x_embedding1)
        mnist_test_acc1 += (pred1.argmax(-1) == source_y).sum().item()
        source_x_embedding2 = encoder2(source_x)
        pred2 = CNet2(source_x_embedding2)
        mnist_test_acc2 += (pred2.argmax(-1) == source_y).sum().item()

    mnist_test_acc1 = mnist_test_acc1 / num_datas
    mnist_test_acc1_.append(mnist_test_acc1)
    mnist_test_acc2 = mnist_test_acc2 / num_datas
    mnist_test_acc2_.append(mnist_test_acc2)

    # save weights
    if epoch % args.model_save_period == 0:
        torch.save(encoder1.state_dict(), args.save_path+ '/encoder1_%i.t7'%(epoch+1))
        torch.save(CNet1.state_dict(), args.save_path+ '/CNet1_%i.t7'%(epoch+1))
        torch.save(encoder2.state_dict(), args.save_path+ '/encoder2_%i.t7'%(epoch+1))
        torch.save(CNet2.state_dict(), args.save_path+ '/CNet2_%i.t7'%(epoch+1))


    logger.info('Epochs %i: source train acc1: %f; source train acc2: %f; source test acc1: %f; source test acc2: %f; mnist test acc1: %f; mnist test acc2: %f;'%(epoch+1, train_acc1, train_acc2,source_test_acc1,source_test_acc2,mnist_test_acc1,mnist_test_acc2))

# plot training curve
plt.clf()
x_axis = range(args.epochs)
plt.plot(x_axis, train_acc1_, label = 'svhn_acc= %g'%train_acc1_[-1])
plt.plot(x_axis, svhn_test_acc1_, label = 'svhn_test_acc= %g'%svhn_test_acc1_[-1])
plt.plot(x_axis, train_acc2_, label = 'shuffle_svhn_acc = %g'%train_acc2_[-1])
plt.plot(x_axis, svhn_test_acc2_, label = 'shuffle_svhn_test_acc= %g'%svhn_test_acc2_[-1])
plt.plot(x_axis, mnist_test_acc1_, label = 'mnist_test_acc= %g'%mnist_test_acc1_[-1])
plt.plot(x_axis, mnist_test_acc2_, label = 'shuffle_mnist_test_acc= %g'%mnist_test_acc2_[-1])
plt.legend()
plt.xlabel('Epochs')
plt.title('Shuffled data training - svhn')
plt.savefig('svhn shuffle class only.png')
