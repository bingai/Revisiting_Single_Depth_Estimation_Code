import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
from camera_transform import *
import cv2

class videoDataset(object):
    def __init__(self, filename, transform=None):
        self.frame = transforms.ToPILImage()(filename)  #bai2 must convert into PILImage()
        # self.frame = filename
        self.transform = transform

    def __getitem__(self, idx):       
        # image = Image.open(self.frame)
        image = self.frame
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return int(1)


def readVideo(frame):
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    image_trans = videoDataset(frame,
                        transform=transforms.Compose([
                        Scale([320, 240]),
                        CenterCrop([304, 228]),
                        ToTensor(),                                
                        Normalize(__imagenet_stats['mean'],
                                 __imagenet_stats['std'])
                       ]))
    image = DataLoader(image_trans, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    # print('load video data successful !!!!!!!')
    # print('++++++ image: ', image)
    # print('++++++ image: ', image.frame)
    return image

def main():
    # image = readVideo('./data/demo/img_nyu2.png')
    # composed_transform = transforms.Compose([Scale([320, 240]), 
    #                                          CenterCrop([304, 228]), 
    #                                          ToTensor(),
    #                                          Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #                                         ])

    # composed_transform = transforms.Compose([Scale([320, 240]), 
    #                                          CenterCrop([304, 228]), 
    #                                          ToTensor(),
    #                                          Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #                                         ])

    
    videofile = './data/demo/video/Manhattan.mp4'
    
    cap = cv2.VideoCapture(videofile)
    
    assert cap.isOpened(), 'Cannot capture source'
    while cap.isOpened():
        
        ret, frame = cap.read()
        print('------- ret: ', ret)
        print('-------- image before transformation:', frame)
        print('-------- image before transformation:', frame.shape)
        print('--------max = {}, min = {}'.format(np.amax(frame), np.amin(frame)))
        
        if ret:
            images = readVideo(frame)
            print('+++++++images: ', images)
            for _, image in enumerate(images):
                print('+++ image: ', image)
                print('+++ image: ', image.size())
                print('+++ max = {}, min = {}'.format(torch.max(image), torch.min(image)))
if __name__ == '__main__':
    main()
