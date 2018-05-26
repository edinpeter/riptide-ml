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
from net_50 import Net as Net_50
from net_100 import Net as Net_100
from net_50_3_3_3 import Net as Net_333


classes = ('1','2','3','4','5','6')

dims = 50

cuda = True

trainset = DiceDataset("data/training_noise", train=True, classes=len(classes), class_max=1000, train_percent=0.6, dims=dims)

testset = DiceDataset("data/training_noise", train=False, classes=len(classes), class_max=1000, train_percent=0.6, dims=dims)

print "Train set length: ", len(trainset)
print "Test set length: ", len(testset)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=8)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=True, num_workers=8)

net = Net_333().cuda() if cuda else Net_333()
print net


criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()

optimizer = optim.ASGD(net.parameters(), lr=0.0011)
#m = nn.LogSoftmax()

# Train on noisy, shifty data
for epoch in range(80):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels, filenames = data
        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        
        inputs, labels = Variable(inputs), Variable(labels)
        
        optimizer.zero_grad()
        outputs = net(inputs)
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

# another 20 epochs on data with backgrounds
trainset = DiceDataset("data/dice_snaps", train=True, classes=len(classes), class_max=300, train_percent=1, dims=dims)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=8)

print "Loaded polishing data"

for epoch in range(20):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels, filenames = data
        if cuda:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            Variable(inputs), Variable(labels)
        
        optimizer.zero_grad()
        outputs = net(inputs)
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

if cuda:
    images = images.cuda()
outputs = net(Variable(images))

_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0

for data in testloader:
    images, labels, filenames = data
    if cuda:
        images = images.cuda()
        labels = labels.cuda()
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()


print('Accuracy of the network on the %i test images: %d %%' %  (len(testset), (
    100 * correct / total)))

class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))
for data in testloader:
    images, labels, filenames = data
    if cuda:
        images = images.cuda()
        labels = labels.cuda()
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels.squeeze())

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
