from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
from skimage import io, color
import numpy as np
import types

import models.dcgan as dcgan
import datasets.dataload as dataload

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='UCF101')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--inc', type=int, default=2, help='input image channels')
parser.add_argument('--outc', type=int, default=2, help='output image channels')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--dataSize', type=int, default=10, help='batches get each time')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--frameSize', type=int, default=16, help='3d network depth')
parser.add_argument('--cropSize', type=int, default=64, help='crop image to this size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

dataloader = dataload.dataloader(opt.dataset, opt.dataroot, opt.dataSize, opt.batchSize, opt.frameSize, opt.cropSize)

ngpu = int(opt.ngpu)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 2
condition = 1
n_extra_layers = int(opt.n_extra_layers)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

if opt.cropSize == 128:
    netG = dcgan.DC_3DGAN_G_unet_128(opt.cropSize, opt.frameSize, opt.inc, opt.outc, ngf, ngpu, n_extra_layers)
elif opt.cropSize == 256:
    netG = dcgan.DC_3DGAN_G_unet(opt.cropSize, opt.frameSize, opt.inc, opt.outc, ngf, ngpu, n_extra_layers)
else:
    netG = dcgan.DC_3DGAN_G_unet_64(opt.cropSize, opt.frameSize, opt.inc, opt.outc, ngf, ngpu, n_extra_layers)

    
netG.apply(weights_init)
if opt.netG != '': # load checkpoint if needed
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


netD = dcgan.DC_3DGAN_D(opt.cropSize, opt.frameSize, nc+condition, ndf, ngpu, n_extra_layers)
netD.apply(weights_init)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)


input = torch.FloatTensor(opt.batchSize, nc, opt.frameSize, opt.cropSize, opt.cropSize)
noise = torch.FloatTensor(opt.batchSize, 1, opt.frameSize, opt.cropSize, opt.cropSize)
fixed_noise = torch.FloatTensor(opt.batchSize, 1, opt.frameSize, opt.cropSize, opt.cropSize).normal_(0, 1)

one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD.cuda()
    netG.cuda()
    input = input.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

# setup optimizer
if opt.adam:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)

gen_iterations = 0

for epoch in range(opt.niter):
    dataloader.reset()
    while True:
        data = dataloader.load()
        if type(data) == bool: # iter ended
            break
        #print (len(data))
        i = 0
        while i < len(data):
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update

            # train the discriminator Diters times
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 100
            else:
                Diters = opt.Diters
            j = 0
            while j < Diters and i < len(data):
                j += 1

                # clamp parameters to a cube
                for p in netD.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)
                
                real_cpu = torch.FloatTensor(data[i])
                i += 1

                # train with real
                #real_cpu, _ = data
                netD.zero_grad()
                batch_size = real_cpu.size(0)
                

                if opt.cuda:
                    real_cpu = real_cpu.cuda()
                
                input.resize_as_(real_cpu[:,1:3,:,:,:]).copy_(real_cpu[:,1:3,:,:,:])
                
                inputv = Variable(input)

                errD_real = netD(inputv, Variable(real_cpu[:,0:1,:,:,:]))
                errD_real.backward(one)

                # train with fake
                noise.resize_(real_cpu.size(0), 1, opt.frameSize, opt.cropSize, opt.cropSize).normal_(0, 1)
                noisev = Variable(noise)
                fake = netG(noisev, Variable(real_cpu[:,0:1,:,:,:]))
                inputv = fake
                inputv.detach()
                errD_fake = netD(inputv, Variable(real_cpu[:,0:1,:,:,:]))
                errD_fake.backward(mone)
                errD = errD_real - errD_fake
                optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False # to avoid computation
            netG.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            
            noise.resize_(real_cpu.size(0), 1, opt.frameSize, opt.cropSize, opt.cropSize).normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev, Variable(real_cpu[:,0:1,:,:,:]))
            errG = netD(fake, Variable(real_cpu[:,0:1,:,:,:]))
            errG.backward(one)
            optimizerG.step()
            gen_iterations += 1

            print('[%d/%d][%d/%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                % (epoch, opt.niter, gen_iterations, len(data),
                errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
            if gen_iterations % 500 == 0:
                #vutils.save_image(real_cpu, '{0}/real_samples.png'.format(opt.experiment))
                fake = netG(Variable(fixed_noise, volatile=True), Variable(real_cpu[:,0:1,:,:,:]))
                y = real_cpu[:,0:1,:,:,:].cpu().numpy()
                uv = fake.data.cpu().numpy()
                dataloader.save(3, np.concatenate((y, uv), axis=1), '{0}/fake_samples_{1}_'.format(opt.experiment, gen_iterations))

    # do checkpointing
    torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
    torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))
