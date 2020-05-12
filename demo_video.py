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


def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model
   

def main():
    # choose the backbone architecture:
    is_resnet = True
    is_densenet = False
    is_senet = False
    model = define_model(is_resnet, is_densenet, is_senet)
    model = torch.nn.DataParallel(model).cuda()

    # load the pretrained model:
    if is_resnet:
        checkpoint = torch.load('./pretrained_model/resnet50_checkpoint.pth.tar')
        # model.load_state_dict(torch.load('./pretrained_model/model_resnet'))
    if is_densenet:
        checkpoint = torch.load('./pretrained_model/densenet_checkpoint.pth.tar')
        # model.load_state_dict(torch.load('./pretrained_model/model_densenet'))
    if is_senet:
        checkpoint = torch.load('./pretrained_model/senet_checkpoint.pth.tar')
        # model.load_state_dict(torch.load('./pretrained_model/model_senet'))
    
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    videofile = './data/demo/video/videoOut.avi'
    # videofile = './data/demo/video/Manhattan.mp4'
    # videofile = './data/demo/video/ManhattanMorePedestrian.mp4'

    cap = cv2.VideoCapture(videofile)
    
    assert cap.isOpened(), 'Cannot capture source'
    
    while cap.isOpened():
        
        ret, frame = cap.read()
        print('------- ret: ', ret)
        # print('-------- image before transformation:', frame.shape)
        if ret:
            print('-------- image before transformation:', frame.shape)

            images = loaddata_video.readVideo(frame)
            # print('--------transformed image:', images)
            # print('+++++ image after transformation:', images.shape())

            for _, image in enumerate(images):
                image = torch.Tensor(image).cuda()

            with torch.no_grad(): 
                out = model(image)      # dtype = np.uint16
            print('1 +++++++++++++++++ Tensor output:')
            print('     Tensor Output: {};      max = {},   min = {} '.format(out.shape, torch.max(out), torch.min(out)))
            
            gray = out.view(out.size(2), out.size(3), out.size(1)).data.cpu().numpy()
            # print('0000000 gray.type: ', type(gray))
            print('2 +++++++++++++++++ Gray output:')
            print('     Gray Output: {};    max = {},   min={}'.format(np.shape(gray), np.amax(gray), np.amin(gray)))

            gray = gray*0.229 + 0.485   # transform the depth output back into original value through imageNet mean & std
            print('\n**********************************************************************************************')
            print(' True Value of Depth Map: {};    max = {},   min={}'.format(np.shape(gray), np.amax(gray), np.amin(gray)))
            print('**********************************************************************************************\n')
            
            gray_scaled01 = (gray - np.amin(gray)) / (np.amax(gray) - np.amin(gray))    # scale output into [0, 1]

            gray_cv2 = np.array(gray_scaled01*255, dtype = np.uint8)    # scale [0, 1] into [0, 255] to compile with CV2 format

            # gray_cv2 = (gray.astype(np.float32)*255).astype(np.uint8)
            print('3 +++++++++++++++++ Gray2CV2 output:')
            print('     Gray2CV2 Output: {};    max = {},   min={}'.format(np.shape(gray_cv2), np.amax(gray_cv2), np.amin(gray_cv2)))
            # gray_cv2 = (gray_cv2, gray_cv2, gray_cv2)
            color_cv2 = cv2.applyColorMap(gray_cv2, cv2.COLORMAP_JET)
            print('4 +++++++++++++++++ Color_CV2 output:')
            print('     Color_CV2 Output: {};    max = {},   min={}'.format(np.shape(color_cv2), np.amax(color_cv2), np.amin(color_cv2)))
            
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
            color_cv2 = cv2.resize(color_cv2, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR) # frame is the original image

            image_numpy_horizontal = np.concatenate((frame, color_cv2), axis=1)
            
            cv2.imshow("depth estimation", image_numpy_horizontal)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('x'):
                break
        else:
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()