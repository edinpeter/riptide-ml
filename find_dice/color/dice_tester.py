import cv2
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import sys
cuda = False

class Candidate():
    def __init__(self, x, y, dim):
        self.x = x
        self.y = y
        self.images = list()
        self.classification = -1
        self.dim = dim
        self.confidence = -1
    def add_image(self, image):
        image = cv2.resize(image, (50, 50))
        image = torch.from_numpy(image).float()
        self.images.append(image)

class Tester():
    model = None

    def __init__(self):
        try:
            self.model = torch.load('models/14_30_30_nc.pt')
        except:
            raise Exception("Cuda / non-cuda model specified on non-cuda / cuda device")
            sys.exit(0)

        if cuda:
            self.model = self.model.cuda()
        self.model.eval()
        self.soft = nn.Softmax(1)

    def test_candidate(self, candidate):
        running_outputs = None
        for c_img in candidate.images:
            c_img = c_img.unsqueeze(0)
            c_img = c_img.transpose(1,3)
            if cuda:
                c_img = c_img.cuda()
            img = Variable(c_img)
            outputs = self.model(img)
            softed = self.soft(Variable(outputs.data))
            running_outputs = running_outputs + softed if running_outputs is not None else softed
            _, predicted = torch.max(softed.data, 1)
        confidence, predicted = torch.max(running_outputs, 1)
        print running_outputs
        print len(candidate.images)
        return confidence, predicted
