import argparse
import torch
import torch.nn as nn
import torch.nn.parallel

from models import modules, net, resnet, densenet, senet

import loaddata
import util
import numpy as np
import sobel


def main():

    # model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    # model = torch.nn.DataParallel(model).cuda()
    # model.load_state_dict(torch.load('./pretrained_model/model_senet'))
    # choose the backbone architecture:
    
    is_resnet = False
    is_densenet = False
    is_senet = True
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

    # test_loader = loaddata.getTestingData(4)
    test_loader = loaddata.getTestingData(1)
    # print('-----------', test_loader)
    test(test_loader, model, 0.25)


def test(test_loader, model, thre):
    model.eval()
    # print('-----------', test_loader)

    totalNumber = 0

    Ae = 0
    Pe = 0
    Re = 0
    Fe = 0

    errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

    for i, sample_batched in enumerate(test_loader):
        image, depth = sample_batched['image'], sample_batched['depth']

        depth = depth.cuda(async=True)
        image = image.cuda()
        # print('------- depth: ', depth.size())
        # print('------- image: ', image.size())
        image = torch.autograd.Variable(image, volatile=True)   # volatile=True tells PyTorch to not bother keeping track of the computation graph. 
        depth = torch.autograd.Variable(depth, volatile=True)
 
        output = model(image)
        # print('----++++ output: ', output.size())

        output = torch.nn.functional.upsample(output, size=[depth.size(2),depth.size(3)], mode='bilinear')
        # print('++++++++ output: ', output.size())

        depth_edge = edge_detection(depth)
        # print('++++++++ output: ', depth_edge.size())
        output_edge = edge_detection(output)
        # print('++++++++ output: ', output_edge.size())

        batchSize = depth.size(0)
        totalNumber = totalNumber + batchSize
        errors = util.evaluateError(output, depth)
        errorSum = util.addErrors(errorSum, errors, batchSize)
        averageError = util.averageErrors(errorSum, totalNumber)

        edge1_valid = (depth_edge > thre)
        edge2_valid = (output_edge > thre)

        nvalid = np.sum(torch.eq(edge1_valid, edge2_valid).float().data.cpu().numpy())
        A = nvalid / (depth.size(2)*depth.size(3))

        nvalid2 = np.sum(((edge1_valid + edge2_valid) ==2).float().data.cpu().numpy())
        P = nvalid2 / (np.sum(edge2_valid.data.cpu().numpy()))
        R = nvalid2 / (np.sum(edge1_valid.data.cpu().numpy()))

        F = (2 * P * R) / (P + R)

        Ae += A
        Pe += P
        Re += R
        Fe += F
        # print('------------', i)

    Av = Ae / totalNumber
    Pv = Pe / totalNumber
    Rv = Re / totalNumber
    Fv = Fe / totalNumber
    print('PV', Pv)
    print('RV', Rv)
    print('FV', Fv)

    averageError['RMSE'] = np.sqrt(averageError['MSE'])
    print(averageError)

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
   

def edge_detection(depth):
    get_edge = sobel.Sobel().cuda()

    edge_xy = get_edge(depth)
    edge_sobel = torch.pow(edge_xy[:, 0, :, :], 2) + \
        torch.pow(edge_xy[:, 1, :, :], 2)
    edge_sobel = torch.sqrt(edge_sobel)

    return edge_sobel


if __name__ == '__main__':
    main()
