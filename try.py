import argparse
import torch
import torch.nn.parallel
import cv2

from models import modules, net, resnet, densenet, senet
import numpy as np
import loaddata_video
import pdb

import matplotlib.image
import matplotlib.pyplot as plt
plt.set_cmap("jet")

def main():
    path = './data/demo/output_camera.png'
    output = cv2.imread(path)
    cv2.imshow("frame", output)
    key = cv2.waitKey(1000)
    # if key & 0xFF == ord('x'):
    #     break

        
if __name__ == '__main__':
    main()
