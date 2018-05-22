import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform, color
import matplotlib.pyplot as plt
import cv2

class DiceImageDataset(Dataset):
    """ Dice Dataset """

    def __init__(self, image_dir, classes, class_max=100, transform=None, dims=100):  

        self.image_dir = image_dir
        self.dims = dims
        self.dice_imgs = list()
        dice_imgs_temp = list()
        for i in range(0, classes):
            self.dice_imgs.append(filter((lambda s: (str(i + 1) + '_') in s) , os.listdir(image_dir)))
            self.dice_imgs[i] = self.dice_imgs[i][0:min(class_max, len(self.dice_imgs[i]))]
            for img in self.dice_imgs[i]:
                dice_imgs_temp.append(img)
        self.dice_imgs = dice_imgs_temp

    def __len__(self):
        return len(self.dice_imgs)

    def __getitem__(self, id):
        img_name = os.path.join(self.image_dir, self.dice_imgs[id])
        image = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (self.dims, self.dims))
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)        
        return image, int(self.dice_imgs[id][0]), self.dice_imgs[id]

    def filename(self, id):
        return self.dice_imgs[id]

if __name__ == "__main__":

    d = DiceImageDataset("/home/peter/Desktop/ftfd/dice_classifier/data/")
    print d[2]
    print 'filename: ', d.filename(2)
    #plt.imshow(d[2][1])
    #plt.show()

