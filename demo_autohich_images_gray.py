import argparse
import torch
import torch.nn.parallel
import torch.nn.functional as F

from models import modules, net, resnet, densenet, senet
import numpy as np
import loaddata_autohich_Gray
import pdb

import matplotlib.image
import matplotlib.pyplot as plt
import os
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

    imagesPath = './data/Autohich/gray_image/'
    image_loader = loaddata_autohich_RGB.readImage(imagesPath)
    # print('-----++++transformed image:', type(image_loader))
    # test(image_loader, model)

    times = []
    for i, [image_name, image] in enumerate(image_loader): 

        t1 = time.time()
        print('----------- i', i)  
        print('---Input info:', image.size(), image_name)
        img_name = image_name[0]
        print('---Input name:', img_name)

        # get the original image for display&scale purposes
        image_original_path = os.path.join(imagesPath, img_name)    
        image_original = matplotlib.pyplot.imread(image_original_path)  #(H, W) = (800, 1280)

        # # image = torch.autograd.Variable(image, volatile=True).cuda()  #bai2 old way
        # # torch.set_grad_enabled(False)   #bai2 new way
        image = torch.Tensor(image).cuda() #bai2 new way
        image_RGB = image.view(image.size(2),image.size(3),image.size(1)).data.cpu().numpy()

        with torch.no_grad():
            out = model(image)
        print('---Output info:', out.size())

        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]
        # inference_time = sum(times)/len(times)*1000
        inference_time = (t2-t1)*1000

        # out = F.upsample(out, size=(out.size(2)*2, out.size(3)*2), mode='bilinear')
        out = F.upsample(out, size=(np.shape(image_original)[0], np.shape(image_original)[1]), mode='bilinear')
        output = out.view(out.size(2),out.size(3)).data.cpu().numpy()
        
        gray = output*0.229 + 0.485   # transform the depth output back into original value through imageNet mean & std
        print('\n**********************************************************************************************')
        print(' True Value of Depth Map: {};    max = {},   min={}'.format(np.shape(gray), np.amax(gray), np.amin(gray)))
        print(' Inference time = {:.2f}ms'.format(inference_time))
        print('**********************************************************************************************\n')

        # fig, axs = plt.subplots(1,2)
        # axs[0].imshow(image_original, cmap='gray')
        # axs[0].set_title(img_name)
        # axs1 = axs[1].imshow(output, cmap='jet')
        # axs[1].set_title('depth_'+ img_name)
        # cbar1 = fig.colorbar(axs1, ax=axs[1], fraction=0.035, pad=0.035)
        # cbar1.minorticks_on()
        # plt.show()

        ##########################################################################
        ###### add the converted RGB image into the figure 
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(image_original, cmap='gray')
        axs[0].set_title(img_name)
        
        axs1 = axs[1].imshow(output, cmap='jet')
        axs[1].set_title('depth_'+ img_name)
        cbar1 = fig.colorbar(axs1, ax=axs[1], fraction=0.035, pad=0.035)
        cbar1.minorticks_on()
         
        axs[2].imshow(image_original)
        axs[2].set_title('RGB_' + img_name)
        plt.show()
        ###### END of add the converted RGB image into the figure 
        ##########################################################################

        
        # # outputPath = os.path.join(imagesPath, img_name)
        # outputPath = imagesPath + 'depth_'+ img_name
        
        # matplotlib.image.imsave(outputPath, output)




# def test(image_loader, model):
#     depthPath = './data/demo/images/'
#     for i, [image_name, image] in enumerate(image_loader): 
#         print('------- i', i)  
#         print('--------Input information:', image.size(), image_name)

#         # # image = torch.autograd.Variable(image, volatile=True).cuda()  #bai2 old way
#         # # torch.set_grad_enabled(False)   #bai2 new way
#         image = torch.Tensor(image).cuda() #bai2 new way

#         # with torch.no_grad():
#         #     out = model(image)
#         # print('--------Output information:', out.size())
#         # output = out.view(out.size(2),out.size(3)).data.cpu().numpy()
#         # outputPath = os.path.join(depthPath, image_name)
#         # print("!!!!!!!!!!", outputPath)
#         # matplotlib.image.imsave(outputPath, output)
        
if __name__ == '__main__':
    main()
