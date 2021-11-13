from model import Net
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms

import argparse
parser = argparse.ArgumentParser(description='easycnn')
parser.add_argument('--epoch', default=2, type=int)
args = parser.parse_args()

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=False,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)#shuffle打乱数据

net=Net()
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

def train():
    for epoch in range(args.epoch):
        running_loss=0.0
        #按批次迭代训练模型
        for i,data in enumerate(trainloader,0):
            inputs,labels=data
            optimizer.zero_grad()#第一步将梯度清零
            outputs=net(inputs)#第二步将输入图像输入网络中，得到输出张量
            loss=criterion(outputs,labels)
            loss.backward()#进行反向传播和梯度更新
            optimizer.step()
            running_loss+=loss.item()
            if (i+1)%2000==0:
                print('[%d,%5d] loss:%.3f'%(epoch+1,i+1,running_loss/2000))
                running_loss=0.0
    print('Finished Training.')

    #设定模型保存位置
    PATH='./cifar_net.pth'
    torch.save(net.state_dict(),PATH)
    
if __name__=="__main__":
    train()