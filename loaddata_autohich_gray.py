import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
from image_transform import *
import os

class imageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_dir, transform=None):

        self.image_dir = image_dir # what directory are the images in
        self.transform = transform

        self.image_names = os.listdir(image_dir) # list all files in the image folder
        self.image_names.sort() # order the images alphabetically
        self.image_names_withPath = [os.path.join(image_dir, image_name) for image_name in self.image_names] # join folder and file name

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_name_withPath = self.image_names_withPath[idx] # get the path of the image at that index
        # image = Image.open(image_name_withPath) # open the image using the path
        image = Image.open(image_name_withPath).convert('RGB')  # open the gray image using the path and convert it into RGB format
        if self.transform:
            image = self.transform(image)
        return image_name, image

    def __len__(self):
        return len(self.image_names)

def readImage(image_file):
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    image_trans = imageDataset(image_file,
                            transform=transforms.Compose([
                                Scale([320, 240]),
                                CenterCrop([304, 228]),
                                ToTensor(),
                                Normalize(__imagenet_stats['mean'],
                                            __imagenet_stats['std'])
                            ]))
    images = DataLoader(image_trans, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    # print('load images successfully !!!!!!!')
    # print('++++++ image_names: ', image_names)
    # print('++++++ image_data: ', type(images))

    return images

def main():

    imagePath = './data/demo/images/'
    image_loader = readImage(imagePath)
    print(type(image_loader))

    for i, [image_name, image] in enumerate(image_loader): 
        print('------- i', i)  
        print('--------Input info:', image.size(), image_name)
        print(image_name)
        image = torch.Tensor(image).cuda() #bai2 new way
        print('--------Input information:', image.size())

    # for image_name, image in image_loader: 
    #     print('--------Input info:', image.size(), type(image_name))
    #     print(image_name)
    #     image = torch.Tensor(image).cuda() #bai2 new way

# if __name__ == '__main__':
#     main()