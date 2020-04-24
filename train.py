import argparse

import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import loaddata
import util
import numpy as np
import sobel
from models import modules, net, resnet, densenet, senet

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--epochs', default=20, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')


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
    global args
    args = parser.parse_args()
    model = define_model(is_resnet=False, is_densenet=True, is_senet=False)
 
    if torch.cuda.device_count() == 8:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda()
        batch_size = 64
    elif torch.cuda.device_count() == 4:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
        batch_size = 32
    else:
        model = model.cuda()
        batch_size = 1
        
    cudnn.benchmark = True  # bai2: This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    train_loader = loaddata.getTrainingData(batch_size)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, optimizer, epoch)
    
    save_checkpoint({'state_dict': model.state_dict()})

def train(train_loader, model, optimizer, epoch):
    criterion = nn.L1Loss()
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    cos = nn.CosineSimilarity(dim=1, eps=0)
    get_gradient = sobel.Sobel().cuda()

    end = time.time()
    for i, sample_batched in enumerate(train_loader):
        image, depth = sample_batched['image'], sample_batched['depth']
        # print('--------Input information:')
        # print('image0 = {0},  image1 = {1},  image2 = {2},  image3 = {3}'.format(image.size(0), image.size(1), image.size(2), image.size(3)))

        depth = depth.cuda(async=True)  #  This can be used to overlap data transfers with computation.
        image = image.cuda()
        image = torch.autograd.Variable(image)  # torch.autograd provides classes and functions implementing automatic differentiation of arbitrary scalar valued functions. 
        depth = torch.autograd.Variable(depth)

        ones = torch.ones(depth.size(0), 1, depth.size(2),depth.size(3)).float().cuda() # batchSize = depth.size(0)
        ones = torch.autograd.Variable(ones)
        optimizer.zero_grad()   # we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
                                # the default action is to accumulate (i.e. sum) the gradients on every loss.backward() call.

        output = model(image)
        # print('--------Output information:')
        # print('output0 = {0},  output1 = {1},  output2 = {2},  output3 = {3}'.format(output.size(0), output.size(1), output.size(2), output.size(3)))

        depth_grad = get_gradient(depth)
        output_grad = get_gradient(output)
        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)      # depth_grad.shape = (-1, 2, x.size(2), x.size(3))
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)

        # print('--------Output_grad_dx information:')
        # print('output_grad_dx0 = {0},  output_grad_dx1 = {1},  output_grad_dx2 = {2},  output_grad_dx3 = {3}'.format(output_grad_dx.size(0), output_grad_dx.size(1), output_grad_dx.size(2), output_grad_dx.size(3)))

        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)
        
        # print('--------output_normal information:')
        # print('output_normal0 = {0},  output_normal1 = {1},  output_normal2 = {2},  output_normal3 = {3}'.format(output_normal.size(0), output_normal.size(1), output_normal.size(2), output_normal.size(3)))

        # depth_normal = F.normalize(depth_normal, p=2, dim=1)
        # output_normal = F.normalize(output_normal, p=2, dim=1)

        loss_depth = torch.log(torch.abs(output - depth) + 0.5).mean()
        loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
        loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
        loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()
        # print('--------loss information:')
        # print('loss_depth.info = ', loss_depth)
        # print('loss_dx.info = ', loss_dx)
        # print('loss_dy.info = ', loss_dy)
        # print('loss_normal.info = ', loss_normal)
        
        loss = loss_depth + loss_normal + (loss_dx + loss_dy)
        # print('loss.info = ', loss)

        # print('+++++++')
        # print('loss.data[0] = ', loss.data)
        # print('loss.data[0].item = ', loss.item)

        # print('image.size(0) = ', image.size(0))
        # losses.update(loss.data[0], image.size(0))  # bai2 old PyTorch=0.4.1
        losses.update(loss.data, image.size(0))  # bai2 new PyTorch>=0.5

        loss.backward() # The graph is used by loss.backward() to compute gradients.
        optimizer.step()    # Performs a parameter update based on the current gradient (stored in .grad attribute of a parameter) and the update rule

        batch_time.update(time.time() - end)
        end = time.time()
   
        batchSize = depth.size(0)

        print('Epoch: [{0}][{1}/{2}]\t'
          'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})'
          .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses))
 

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 5))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


if __name__ == '__main__':
    main()
