import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

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

model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=False)
criterion = nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
model.eval()

PATH= './alexnet.pth'

# for epoch in range(2):
#     running_loss=0
#     for i, data in enumerate(trainloader,0):
#         inputs,labels=data
#         optimizer.zero_grad()
#         outputs=model(inputs)
#         loss=criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss+=loss.item()
#         print(i)

# torch.save(model.state_dict(),PATH)

dataiter = iter(testloader)
images, labels = dataiter.next()

import matplotlib.pyplot as plt 
import numpy as np 
def imshow(img): 
    img=img/2+0.2
    npimg =img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
model.load_state_dict(torch.load(PATH))

outputs=model(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))


# total=0
# correct=0
# with torch.no_grad():
#     for data in testloader:
#         images,labels= data
#         outputs = model(images)
#         _,predicted=torch.max(outputs.data,1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         print('hello')

# print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
