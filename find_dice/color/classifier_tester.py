import sys
sys.path.append('dataset_classes/')
import torch
from dice_dataset import DiceDataset
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import random
from net_50 import Net as Net_50
from net_100 import Net as Net_100


verbose = True

classes = ('1','2','3','4','5','6')
dims = 50

data_start = time.time()
testset = DiceDataset("data/training_noise", train=False, classes=len(classes), class_max=2000, train_percent=0, dims=dims)
data_end = time.time()

model_start = time.time()
model = torch.load('models/93_8000.pt').cuda()
model.eval()
model_end = time.time()

print "Len: ", len(testset)

total = 0
correct = 0
start = time.time()
times = list()
for i in range(0, 500):
    start_sample = time.time()
    r1 = random.randint(0, len(testset) - 1)
    r2 = random.randint(0, len(testset) - 1)
    r3 = random.randint(0, len(testset) - 1)
    r4 = random.randint(0, len(testset) - 1)
    samples = [r1, r2, r3, r4]
    #print r1, r2, r3, r4, len(testset) - 1
    t = torch.cat((testset[r1][0].unsqueeze(0), testset[r2][0].unsqueeze(0), testset[r3][0].unsqueeze(0), testset[r4][0].unsqueeze(0)), dim=0)
    soft = nn.Softmax(1)
    optim = Variable(t.cuda())

    outputs = model(optim)
    #print "Testing..."
    _, predicted = torch.max(outputs.data, 1)
    if verbose:
        #print outputs.data
        #print soft(Variable(outputs.data))
        #print outputs.data[1]
        pass
    softed = soft(Variable(outputs.data))
    for i in range(0,4):
        total = total + 1
        if verbose:
            if softed.data[i][predicted[i]] > 0.5 and softed.data[i][predicted[i]] < 1:
                print "Confidence %1.4f" % (softed[i][predicted[i]])
                print "Prediction: %i" % (predicted[i] + 1)
                print "Actual: %10s" % (testset[samples[i]][2])
                print "\n\n"

        if str(predicted[i] + 1)+'_' in testset[samples[i]][2]:
                correct = correct + 1
    #end = time.clock()
    end_sample = time.time()
    times.append(end_sample - start_sample)
end = time.time()

print "Data loaded in: %2.2f seconds" % (data_end - data_start)
print "Model loaded in: %2.2f seconds" % (model_end - model_start)
print "Completed in: %1.5f seconds: " % (end - start)
print "\tAverage time per frame: %2.6f" % (sum(times) / (float(len(times)) * 4.0))
print "Net accuracy: %2.2f %%" % (100 * float(correct) / float(total))
print "Correct samples: %i" % (correct)
print "Total samples: %i" % (total)
