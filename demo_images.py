import argparse
import torch
import torch.nn.parallel
import torch.nn.functional as F

from models import modules, net, resnet, densenet, senet
import numpy as np
import loaddata_images
import pdb

import matplotlib.image
import matplotlib.pyplot as plt
plt.set_cmap("jet")
import os

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
    is_resnet = False
    is_densenet = True
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

    imagesPath = './data/demo/images/'
    image_loader = loaddata_images.readImage(imagesPath)
    # print('-----++++transformed image:', type(image_loader))
    # test(image_loader, model)

    for i, [image_name, image] in enumerate(image_loader): 
        print('----------- i', i)  
        print('---Input info:', image.size(), image_name)
        img_name = image_name[0]
        print('---Input mane:', img_name)


        # # image = torch.autograd.Variable(image, volatile=True).cuda()  #bai2 old way
        # # torch.set_grad_enabled(False)   #bai2 new way
        image = torch.Tensor(image).cuda() #bai2 new way

        with torch.no_grad():
            out = model(image)
        print('---Output info:', out.size())

        # out = F.upsample(out, size=(out.size(2)*2, out.size(3)*2), mode='bilinear')
        output = out.view(out.size(2),out.size(3)).data.cpu().numpy()
        
        # outputPath = os.path.join(imagesPath, img_name)
        outputPath = imagesPath + 'depth_'+ img_name
        # print("!!!!!!!!!!", outputPath)
        
        matplotlib.image.imsave(outputPath, output)


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
