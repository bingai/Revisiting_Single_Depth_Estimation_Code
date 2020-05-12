import argparse
import torch
import torch.nn.parallel
import cv2

from models import modules, net, resnet, densenet, senet
import numpy as np
import loaddata_camera
import pdb

import matplotlib.image
import matplotlib.pyplot as plt
plt.set_cmap("jet")
import time

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

    # videofile = './data/demo/video/Manhattan.mp4'
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 2000)
    
    assert cap.isOpened(), 'Cannot capture source'
    
    times = []
    while cap.isOpened():
        
        ret, frame = cap.read()
        print('------- ret: ', ret)
        print('-------- image before transformation:', frame.shape)
        if ret:
            print('-------- image before transformation:', frame.shape)
            
            images = loaddata_camera.readVideo(frame)
            # print('--------transformed image:', images)
            # print('+++++ image after transformation:', images.shape())

            for _, image in enumerate(images):
                image = torch.Tensor(image).cuda()

            t1 = time.time()
            # print('!!!!!!! t1 = ', t1)
            with torch.no_grad(): 
                out = model(image)      # dtype = np.uint16
                print('+++++++++++++++++ output before transformation:')
            print(' Output: ', out.size())
            print(' Output max = {}, min = {} '.format(torch.max(out), torch.min(out)))

            t2 = time.time()
            times.append(t2-t1)
            times = times[-20:]
            print('++++++ t2 = ', t2)


            # print('     !!!!!!!!!!!!!!!!    Inference time: {:.2f}ms'.format(t2-t1))
            print('     !!!!!!!!!!!!!!!!    Inference time: {:.2f}ms'.format(sum(times)/len(times)*1000))

            gray = out.view(out.size(2), out.size(3)).data.cpu().numpy()
            # gray = out.view(out.size(2), out.size(3), out.size(1)).data.cpu().numpy()
            print('0000000 gray.type: ', type(gray))
            # output = cv2.merge((gray, gray, gray))
            output = gray
            # color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            # output = cv2.convertScaleAbs(color, 255)

            print(' output gray2color: ', output.shape)  
            print(' max distance is ={}, min distance is = {} '.format(np.amax(output), np.amin(output)))
            matplotlib.image.imsave('./data/demo/output_camera.png', output)  # store value [0,1]
            # cv2.imwrite('./data/demo/output_camera.png', output)
            # print('------ output: ', output) 
            print('------ ++++++++++++++ output after transformation: ')
            print(' output after transformation: ', output.shape)  
            print(' max distance is ={}, min distance is = {} '.format(np.amax(output), np.amin(output)))
            # output = cv2.imread('./data/demo/output_camera.png', cv2.IMREAD_UNCHANGED)
            output = cv2.imread('./data/demo/output_camera.png')

            # print('------ output: ', output) 
            print('++++++++++++++++++++++ output final:')
            print(' output : ', output.shape)  
            print(' max distance is ={}, min distance is = {} '.format(np.amax(output), np.amin(output)))

            # cv2.imshow("frame", output)

            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)

            # frame = cv2.resize(frame, (533, 400), interpolation=cv2.INTER_LINEAR)

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
