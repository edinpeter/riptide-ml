import sys
sys.path.append('dataset_classes/')
import torch
import torchvision
import torchvision.transforms as transforms
from dice_dataset import DiceDataset
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common_net import Net

classes = ('1','2','3','4','5','6')

dims = 50

trainset = DiceDataset("data/training_noise", train=True, classes=len(classes), class_max=5000, train_percent=0.4, dims=dims)

testset = DiceDataset("data/training_noise", train=False, classes=len(classes), class_max=5000, train_percent=0.4, dims=dims)

print "Train set length: ", len(trainset)
print "Test set length: ", len(testset)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=8)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=True, num_workers=8)

net = Net(dims, classes).cuda()


criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()

optimizer = optim.ASGD(net.parameters(), lr=0.0011)
#m = nn.LogSoftmax()

for epoch in range(150):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels, filenames = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        
        #print inputs

        optimizer.zero_grad()
        outputs = net(inputs)
        #print outputs
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if i % 50 == 0:
            print('[%d, %5d] loss: %.8f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    if running_loss < 0.002:
        optimizer = optim.ASGD(net.parameters(), lr=0.0001)
    if running_loss < 0.0002:
        optimizer = optim.ASGD(net.parameters(), lr=0.00001)


print('Finished Training')

dataiter = iter(testloader)
images, labels, filenames = dataiter.next()

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(Variable(images.cuda()))

_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0

for data in testloader:
    images, labels, filenames = data
    outputs = net(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('Accuracy of the network on the %i test images: %d %%' %  (len(testset), (
    100 * correct / total)))

class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))
for data in testloader:
    images, labels, filenames = data
    outputs = net(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels.cuda()).squeeze()

    for i in range(len(labels)):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

print class_correct
print class_total

for i in range(len(classes)):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

torch.save(net, 'classifier_cuda.pt')
