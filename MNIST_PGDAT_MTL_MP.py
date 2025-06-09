#----------------------------------------------------------------------------------------------------------------------------------------
# MNIST Adversarial Training using Mixed Perturbation
# by C.H.Hyun
# June. 2025
# Description:
#   - Provides only the essential components of the adversarial training pipeline with Mixed Perturbation(MP).
#   - Users can customize it freely.
#   - For example, someone might want to add a learning rate scheduler, or change output dimensions to suit their specific dataset. 
#   - This code has been tested with the following environment. Compatibility with other versions is not guaranteed, and any issues must be resolved by the user.
#      - Python 3.8  
#      - Pytorch 2.1.2
#      - torch==1.8.0+cu111  
#      - torchvision==1.8.0+cu111 
#      - torchsummary==1.5.1
#      - advertorch==0.2.4  
#----------------------------------------------------------------------------------------------------------------------------------------

import time
import random
import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchsummary import summary
from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.attacks import LinfPGDAttack


# define model 
class LeNet5Madry(nn.Module):
    # This LeNet model is based on the CNN architecture used by Madry et al. in their adversarial training paper -> https://arxiv.org/pdf/1706.06083
    def __init__(
            self, nb_filters=(1, 32, 64), kernel_sizes=(5, 5),
            paddings=(2, 2), strides=(1, 1), pool_sizes=(2, 2),
            nb_hiddens=(7 * 7 * 64, 1024), nb_classes_1 = 10, nb_classes_2 = 2):
        super(LeNet5Madry, self).__init__()
        self.conv1 = nn.Conv2d(
            nb_filters[0], nb_filters[1], kernel_size=kernel_sizes[0],
            padding=paddings[0], stride=strides[0])
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(pool_sizes[0])
        self.conv2 = nn.Conv2d(
            nb_filters[1], nb_filters[2], kernel_size=kernel_sizes[1],
            padding=paddings[0], stride=strides[0])
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(pool_sizes[1])
        self.linear1 = nn.Linear(nb_hiddens[0], nb_hiddens[1])
        self.relu3 = nn.ReLU(inplace=True)
        self.outlayer1 = nn.Linear(nb_hiddens[1], nb_classes_1)
        self.outlayer2 = nn.Linear(nb_hiddens[1], nb_classes_2)
    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out1 = self.outlayer1(out)
        out2 = self.outlayer2(out)
        return out1, out2

# module that outputs only 'first logit'
class SelectOutput1(nn.Module):
    def __init__(self):
        super(SelectOutput1, self).__init__()
    def forward(self,x):
        out = x[0]
        return out

# module that outputs only 'second logit'
class SelectOutput2(nn.Module):
    def __init__(self):
        super(SelectOutput2, self).__init__()
    def forward(self,x):
        out = x[1]
        return out

# random sampling perturbation weight
def sample_pwr(alpha0):
    a0 = random.uniform(alpha0, 1.0) # perturbation weight for the main task
    a1 = 1 - a0                      # perturbation weight for the auxiliary task
    return a0, a1

# main code
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PGDAT & MTL on CIFAR10')
    parser.add_argument('--seed', default = 0, type = int)
    parser.add_argument('--epsilon', default = 0.3, type = float)
    parser.add_argument('--capacity', default = 16, type = int)
    parser.add_argument('--class1', default = 0, type = int)      # select classes => 0 = id, 1 = oe, 2 = pm, 3 = cv, 4 = ci, 5 = mt
    parser.add_argument('--class2', default = 1, type = int)
    parser.add_argument('--traintask', default = "maintask", help = "maintask | subtask")
    parser.add_argument('--testtask', default = "maintask", help = "maintask | subtask")
    parser.add_argument('--alpha0', default = 5, type = float, help = "minimum of main task perturbation weight")
    args = parser.parse_args()
    capacity = args.capacity
    epsilon = args.epsilon
    seed = args.seed
    class1 = args.class1
    class2 = args.class2
    traintask = args.traintask
    testtask = args.testtask
    alpha0 = args.alpha0
    alpha0 = alpha0/10.0


    # seed setting
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

    # model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LeNet5Madry(nb_filters=(1, 2*capacity, 4*capacity), nb_hiddens=(7 * 7 * 4*capacity, 64*capacity))
    print(model)
    model.to(device) #model = model.to('cuda')
    print(next(model.parameters()).device)
    summary(model, input_size=(1, 28, 28))
    # define model for generating adversary 
    new_model1 = nn.Sequential(
        model,
        SelectOutput1() ###!
    )
    new_model2 = nn.Sequential(
        model,
        SelectOutput2() ###!
    )

    # set hyperparameters - Revise the code if users want these hyperparameters be included in parser
    train_batch_size = 50
    test_batch_size = 1000
    log_interval = 200
    nb_epoch = 200
    learning_rate = 1e-4 # 1e-4 | 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    criterion = torch.nn.CrossEntropyLoss().to(device) # softmax included


    # load data
    Train_data = torch.load('./Data/MNIST/Traindata_MNIST.pt') 
    Train_label = torch.load('./Data/MNIST/Trainlabel_MNIST.pt')
    Test_data = torch.load('./Data/MNIST/Testdata_MNIST.pt')
    Test_label = torch.load('./Data/MNIST/Testlabel_MNIST.pt')

    # get labels for each task
    # index 0 = identity, 1 = odd-even, 2 = prime number
    Train_label1 = Train_label[class1].long()
    Train_label2 = Train_label[class2].long()
    Test_label1 = Test_label[class1].long()
    Test_label2 = Test_label[class2].long()

    # make data loader
    trn_dataset = TensorDataset(Train_data, Train_label1, Train_label2)
    trn_dataloader = DataLoader(trn_dataset, batch_size = train_batch_size, shuffle = True)
    tst_dataset = TensorDataset(Test_data, Test_label1, Test_label2)
    tst_dataloader = DataLoader(tst_dataset, batch_size = test_batch_size, shuffle = True)


    # set model save path  
    if args.class1 == 0:
        dirclass1 = "id"
    elif args.class1 == 1:
        dirclass1 = "oe"
    elif args.class1 == 2:
        dirclass1 = "pm"

    if args.class2 == 0:
        dirclass2 = "id"
    elif args.class2 == 1:
        dirclass2 = "oe"
    elif args.class2 == 2:
        dirclass2 = "pm"
    
    dirName0 = './trained_model/MNIST/PGDAT_MTL_MP_{}{}_alpha0({})'.format(dirclass1, dirclass2, alpha0) ###!
    try:
        os.mkdir(dirName0)
        print("Directory " , dirName0 ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName0 ,  " already exists")
    

    total_batch = len(trn_dataloader)
    print('num of total training batch : {}'.format(total_batch))

    # Training
    for epoch in range(nb_epoch):
        model.train()
        trncorrect1 = 0
        trncorrect2 = 0
        avg_loss = 0
        total_trn = 0
        batch_idx = 0
        for data, target1, target2 in trn_dataloader: # train_loader
            batch_time = time.time()
            batch_idx += 1
            data, target1, target2 = data.to(device), target1.to(device), target2.to(device)
            optimizer.zero_grad()
            

            ### -- get Mixed Perturbation --    
            # make perturbation for each task       
            adversary_train1 = LinfPGDAttack(new_model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsilon, nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
            with ctx_noparamgrad_and_eval(model):
                data_perturb1 = adversary_train1.perturb(data, target1)
            adversary_train2 = LinfPGDAttack(new_model2, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsilon, nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
            with ctx_noparamgrad_and_eval(model):
                data_perturb2 = adversary_train2.perturb(data, target2)
            t1pw, t2pw = sample_pwr(alpha0) # get perturbation weights
            data_MP = data + t1pw*(data_perturb1 - data) + t2pw*(data_perturb2 - data) # weighted summation

            # forward using MP
            output1, output2 = model(data_MP)


            loss1 = F.cross_entropy(output1, target1, reduction='elementwise_mean')
            loss2 = F.cross_entropy(output2, target2, reduction='elementwise_mean')
            loss = (loss1 + loss2) / 2
            loss.backward()
            optimizer.step()
            avg_loss += loss / total_batch
            total_trn += target1.size(0)
            _, predicted1 = output1.max(1)
            _, predicted2 = output2.max(1)
            trncorrect1 += predicted1.eq(target1).sum().item()
            trncorrect2 += predicted2.eq(target2).sum().item()

        print('epoch : ', epoch)
        print('training accuarcy of main task : ', 100. * trncorrect1 / total_trn)
        print('training accuarcy of aux task  : ', 100. * trncorrect2 / total_trn)
        print('training loss : ', avg_loss)

        if epoch%10 == 0: # test every 10 epoch
            model.eval()
            test_clnloss1 = 0
            test_clnloss2 = 0
            clncorrect1 = 0
            clncorrect2 = 0
            test_advloss1 = 0
            test_advloss2 = 0
            advcorrect1 = 0
            advcorrect2 = 0
            
            # Test after training
            for clndata, target1, target2 in tst_dataloader:
                clndata, target1, target2 = clndata.to(device), target1.to(device), target2.to(device)

                # test with clean samples
                with torch.no_grad():
                    output1, output2 = model(clndata)
                test_clnloss1 += F.cross_entropy(
                    output1, target1, reduction='sum').item()
                test_clnloss2 += F.cross_entropy(
                    output2, target2, reduction='sum').item()
                pred1 = output1.max(1, keepdim=True)[1]
                clncorrect1 += pred1.eq(target1.view_as(pred1)).sum().item()
                pred2 = output2.max(1, keepdim=True)[1]

                # test with adversarial examples
                if testtask == "maintask":
                    test_model_adversary = new_model1
                    test_target = target1
                elif testtask == "subtask":
                    test_model_adversary = new_model2
                    test_target = target2
                adversary_test = LinfPGDAttack(test_model_adversary, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsilon, nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
                advdata = adversary_test.perturb(clndata, test_target)       
                with torch.no_grad():
                    output1, output2 = model(advdata)
                test_advloss1 += F.cross_entropy(
                    output1, target1, reduction='sum').item()
                test_advloss2 += F.cross_entropy(
                    output2, target2, reduction='sum').item()
                pred1 = output1.max(1, keepdim=True)[1]
                advcorrect1 += pred1.eq(target1.view_as(pred1)).sum().item()
                pred2 = output2.max(1, keepdim=True)[1]
                advcorrect2 += pred2.eq(target2.view_as(pred2)).sum().item()


            test_clnloss1 /= len(tst_dataloader.dataset)
            print('Test set: avg cln loss 1: {:.4f},'
                  ' cln acc 1: ({:.4f}%)'.format(
                      test_clnloss1, 100. * clncorrect1 / len(tst_dataloader.dataset)))
            
            test_clnloss2 /= len(tst_dataloader.dataset)    
            print('Test set: avg cln loss 2: {:.4f},'
                  ' cln acc 2: ({:.4f}%)\n'.format(
                      test_clnloss2, 100. * clncorrect2 / len(tst_dataloader.dataset)))          

            test_advloss1 /= len(tst_dataloader.dataset)
            print('Test set: avg adv loss 1: {:.4f},'
                  ' adv acc 1: ({:.4f}%)'.format(
                      test_advloss1, 100. * advcorrect1 / len(tst_dataloader.dataset)))
            test_advloss2 /= len(tst_dataloader.dataset)
            print('Test set: avg adv loss 2: {:.4f},'
                  ' adv acc 2: ({:.4f}%)\n'.format(
                      test_advloss2, 100. * advcorrect2 / len(tst_dataloader.dataset)))

            save_path = dirName0+'/%d.pt'%(epoch) #!
            torch.save(model, save_path)
