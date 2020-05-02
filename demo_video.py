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
        print('-------- image before transformation:', frame.shape)
        if ret:
            # cv2.imshow("input", frame)
            images = loaddata_video.readVideo(frame)
            # print('--------transformed image:', images)
            # print('+++++ image after transformation:', images.shape)

            for _, image in enumerate(images):
                image = torch.Tensor(image).cuda()

            with torch.no_grad(): 
                out = model(image)
            print('---++++ Output: ', out.size())
            print('---++++ Output max = {}, min = {}: '.format(torch.max(out), torch.min(out)))
            
            output = out.view(out.size(2), out.size(3)).data.cpu().numpy()
            matplotlib.image.imsave('./data/demo/output_camera.png', output)  # store value [0,1]

            print('------ output: ', output.shape)  
            print('------ max distance is ={0}, min distance is = {1} '.format(np.amax(output), np.amin(output)))
            output = cv2.imread('./data/demo/output_camera.png')
            print('------ output: ', output.shape)  
            print('------ max distance is ={0}, min distance is = {1} '.format(np.amax(output), np.amin(output)))
            # cv2.imshow("depth", output)

            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
            # frame = cv2.resize(frame, (800, 450), interpolation=cv2.INTER_LINEAR)

            output_resize = cv2.resize(output, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR) # frame is the original image

            image_numpy_horizontal = np.concatenate((frame, output_resize), axis=1)
            
            cv2.imshow("depth estimation", image_numpy_horizontal)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('x'):
                break
        else:
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()




    # torch.no_grad()

    # video_loader = loaddata_video.readVideo('./data/demo/video/Manhattan.mp4')

    # image = torch.Tensor(image).cuda() #bai2 new way
    # out = model(image)
    # print('--------Output information:')
    # print(out.view(out.size(2),out.size(3)).data.cpu().numpy())
    # matplotlib.image.imsave('./data/demo/out.png', out.view(out.size(2),out.size(3)).data.cpu().numpy())

#     test(nyu2_loader, model)


# def test(nyu2_loader, model):
#     for i, image in enumerate(nyu2_loader):   
#         print('--------Input information:')
#         print('image0 = {0},  image1 = {1},  image2 = {2},  image3 = {3}'.format(image.size(0), image.size(1), image.size(2), image.size(3)))  
#         # image = torch.autograd.Variable(image, volatile=True).cuda()  #bai2 old way
#         # torch.set_grad_enabled(False)   #bai2 new way
#         torch.no_grad()
#         image = torch.Tensor(image).cuda() #bai2 new way
#         # print('--------Input information:')
#         # print('image0 = {0},  image1 = {1},  image2 = {2},  image3 = {3}'.format(image.size(0), image.size(1), image.size(2), image.size(3)))
#         out = model(image)
#         print('--------Output information:')
#         print('out0 = {0},  out1 = {1},  out2 = {2},  out3 = {3}'.format(out.size(0), out.size(1), out.size(2), out.size(3)))
#         print(out.view(out.size(2),out.size(3)).data.cpu().numpy())
#         matplotlib.image.imsave('./data/demo/out.png', out.view(out.size(2),out.size(3)).data.cpu().numpy())
        
if __name__ == '__main__':
    main()
