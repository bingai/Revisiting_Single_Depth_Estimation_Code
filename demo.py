import argparse
import torch
import torch.nn.parallel

from models import modules, net, resnet, densenet, senet
import numpy as np
import loaddata_demo as loaddata
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

    nyu2_loader = loaddata.readNyu2('./data/demo/img_nyu2.png')
    # print('-----++++transformed image:', type(nyu2_loader))
    test(nyu2_loader, model)


def test(nyu2_loader, model):
    # print('!!!!!!!!!-----++++transformed image:', type(nyu2_loader))

    for i, image in enumerate(nyu2_loader): 
        # print('------- i', i)  
        # print('--------Input information:')
        print('image0 = {0},  image1 = {1},  image2 = {2},  image3 = {3}'.format(image.size(0), image.size(1), image.size(2), image.size(3)))  
        # image = torch.autograd.Variable(image, volatile=True).cuda()  #bai2 old way
        # torch.set_grad_enabled(False)   #bai2 new way
        image = torch.Tensor(image).cuda() #bai2 new way
        # print('--------Input information:')
        # print('image0 = {0},  image1 = {1},  image2 = {2},  image3 = {3}'.format(image.size(0), image.size(1), image.size(2), image.size(3)))
        with torch.no_grad():
            out = model(image)
        print('--------Output information:')
        print('out0 = {0},  out1 = {1},  out2 = {2},  out3 = {3}'.format(out.size(0), out.size(1), out.size(2), out.size(3)))
        print(out.view(out.size(2),out.size(3)).data.cpu().numpy())
        matplotlib.image.imsave('./data/demo/out.png', out.view(out.size(2),out.size(3)).data.cpu().numpy())
        matplotlib.image.imshow('./data/demo/out.png', out.view(out.size(2),out.size(3)).data.cpu().numpy())
        
if __name__ == '__main__':
    main()
