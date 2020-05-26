import math

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt 
import numpy as np 
def imshow(img): 
    img=img/2+0.2
    npimg =img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

model = models.alexnet(pretrained=True)
criterion = nn.CrossEntropyLoss()
model.eval()


for params in model.parameters():
    params.requires_grad=False
model.classifier[6]=nn.Linear(4096,10)
optimizer=optim.SGD(model.parameters(),lr=0.001,momentum=0.9)

bound = 1 / math.sqrt(model.classifier[6].weight.size(1))
model.classifier[6].weight.data.uniform_(-bound,bound)
PATH= './alexnet.pth'

for epoch in range(2):
    running_loss=0
    batch_loss=0
    for i, data in enumerate(trainloader,0):
        inputs,labels=data
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        loss.backward()
        if(i%500==499):
            print(batch_loss)
            batch_loss=0
        optimizer.step()
        running_loss+=loss.item()
        batch_loss+=loss.item()
        print(i)
model.eval()     

torch.save(model.state_dict(),PATH)

model.load_state_dict(torch.load(PATH))

total=0
correct=0
with torch.no_grad():
    for data in testloader:
        images,labels= data
        outputs = model(images)
        _,predicted=torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

