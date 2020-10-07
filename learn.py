import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
args = sys.argv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


t1X = np.load("traindata_X.npz")
traindata_X = []
for k in t1X.keys():
    traindata_X = t1X[k]

t1Y = np.load("traindata_Y.npz")
traindata_Y = []
for k in t1Y.keys():
    traindata_Y = t1Y[k]

l1X = np.load("testdata_X.npz")
testdata_X = []
for k in l1X.keys():
    testdata_X = l1X[k]

l1Y = np.load("testdata_Y.npz")
testdata_Y = []
for k in l1Y.keys():
    testdata_Y = l1Y[k]

    
traindata_X = np.array(traindata_X)
traindata_Y = np.array(traindata_Y)
testdata_X = np.array(testdata_X)
testdata_Y = np.array(testdata_Y)


t_X_train = torch.from_numpy(traindata_X).float()
t_Y_train = torch.from_numpy(traindata_Y).float()
t_X_test  = torch.from_numpy(testdata_X ).float()
t_Y_test  = torch.from_numpy(testdata_Y).float()

BATCH_SIZE = 1000

dataset_train = TensorDataset(t_X_train, t_Y_train)
dataset_test = TensorDataset(t_X_test, t_Y_test)

loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
loader_valid = DataLoader(dataset_test, batch_size=1)


import my_model
import generate_path

criterion = nn.BCELoss()

device = 'cuda'
epoch = int(args[1])


model = my_model.Net()
model = model.to(device)

optimizer = optim.Adam(           # 最適化アルゴリム
model.parameters(),          # 最適化で更新対象のパラメーター（重みやバイアス）
    #lr=0.03,            # 更新時の学習率
    weight_decay = 2e-4#L2正則化5e-4よさげ
    ) # L2正則化（※不要な場合は0か省略）


# 学習
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.reshape(-1,1))
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                s = 'Train Epoch: {}/{}\tLoss: {:.6f}'.format(
                    e+1, epoch, loss.item())
                sys.stdout.write("\033[2K\033[G%s" % s)
                sys.stdout.flush()
    print()

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    all_n = 0
    outputs = []
    likelihoods = []
    with torch.no_grad():
        for data, target in test_loader:
            raw_data = data[0]
            data, target = data.to(device), target.to(device)
            output = model(data)
            #print(output)
            #test_loss += criterion(output, target.reshape(-1, 1)).item() # sum up batch loss
            #pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()
            #print(target)
            for i in range(len(output)):
                all_n += 1
                #print(target[i], output[i])
                outputs.append(output.item())
                likelihood = generate_path.which_f(generate_path.f1, generate_path.f2, raw_data).item()
                likelihoods.append(likelihood)
                #print("output =", round(output.item(), 4)," lilelihood =" ,round(generate_path.which_f(generate_path.f1, generate_path.f2, raw_data).item(), 4))
                if output[i] <0.5 and target[i] == 0:
                    correct +=1 
                elif output[i] >0.5 and target[i] == 1:
                    correct +=1
    print("correct rate =",correct/all_n)
    if True:
        plt.scatter(likelihoods, outputs)
        plt.grid()
        plt.xlabel("likelihood")
        plt.ylabel("output")
        plt.savefig("likelihoodoutput.png")
        plt.show()
    


train(model, device, loader_train, optimizer, epoch)
test(model, device, loader_valid)










