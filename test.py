from model import Net
import torch
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--choice', default='', type=str)
args = parser.parse_args()

def test():
    dataiter=iter(testloader)
    images,labels=dataiter.next()
    print('GroundTruth:',' '.join('%5s'%classes[labels[j]] for j in range(4)))

    PATH='./cifar_net.pth'
    net=Net()
    net.load_state_dict(torch.load(PATH))
    outputs=net(images)
    _,predicted=torch.max(outputs,1)
    print('GroundTruth:',' '.join('%5s'%classes[predicted[j]] for j in range(4)))
    correct=0
    total=0
    with torch.no_grad():
        for data in testloader:
            images,labels=data
            outputs=net(images)
            _,predicted=torch.max(outputs.data,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    print('Accuracy of the network on the 10000 test images:%d %%'%(100*correct/total))

def test_mul():
    #分别测试不同类别的模型准确率
    PATH='./cifar_net.pth'
    net=Net()
    net.load_state_dict(torch.load(PATH))
    class_correct=list(0. for i in range(10))
    class_total=list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images,labels=data
            outputs=net(images)
            _,predicted=torch.max(outputs,1)
            c=(predicted==labels).squeeze()
            for i in range(4):
                label=labels[i]
                class_correct[label]+=c[i].item()
                class_total[label]+=1
    for i in range(10):
        print('Accuracy of %5s:%2d %%'%(classes[i],100*class_correct[i]/class_total[i]))
        
if __name__=='__main__':
    import torch
    import torchvision
    import torchvision.transforms as transforms
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=False,transform=transform)
    testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)
    classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
    if args.choice!='mul':
        test()
    else:
        test_mul()
    