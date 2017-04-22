import torch
import torch.nn as nn
import torch.nn.parallel

class DC_3DGAN_D(nn.Module):
    def __init__(self, isize, nframe, nc, ndf, ngpu, n_extra_layers=0):
        super(DC_3DGAN_D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        
        main = nn.Sequential()
        #input is nc x isize x isize
        main.add_module('initial.conv.{0}-{1}'.format(nc, ndf), 
                        nn.Conv3d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial.relu.{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf
        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, cndf),
                            nn.Conv3d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, cndf),
                            nn.BatchNorm3d(cndf))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid.{0}-{1}.conv'.format(in_feat, out_feat),
                            nn.Conv3d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(out_feat),
                            nn.BatchNorm3d(out_feat))
            main.add_module('pyramid.{0}.relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final.{0}-{1}.conv'.format(cndf, 1),
                        nn.Conv3d(cndf, 1, (int(4*nframe/isize), 4, 4), 1, 0, bias=False))
        self.main = main


    def forward(self, input, condition):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        inputs = torch.cat([input, condition], 1)
        output = nn.parallel.data_parallel(self.main, inputs, gpu_ids)
        output = output.mean(0)
        return output.view(1)

        
class DC_3DGAN_G_unet(nn.Module):
    def __init__(self, isize, nframe, inc, outc, ngf, ngpu, n_extra_layers=0):
        super(DC_3DGAN_G_unet, self).__init__()
        self.ngpu = ngpu
        
        cframe = nframe
        assert isize % 16 == 0, "isize has to be a multiple of 16" 
        
        #input is (inc) x nframe x 256 x 256
        self.conv1 = nn.Conv3d(inc, ngf, 4, 2, 1)
        
        self.leakyrelu = nn.LeakyReLU(True)
        self.bn1 = nn.BatchNorm3d(ngf)
        self.bn2 = nn.BatchNorm3d(ngf*2)
        self.bn4 = nn.BatchNorm3d(ngf*4)
        self.bn8 = nn.BatchNorm3d(ngf*8)
        
        cframe /= 2
        
        #input is (ngf) x nframe x 128 x 128
        self.conv2 = nn.Conv3d(ngf, ngf*2, ((cframe>1)+3, 4, 4), 2, 1)
        cframe /= 2
        
        #input is (ngf*2) x nframe x 64 x 64
        self.conv3 = nn.Conv3d(ngf*2, ngf*4, ((cframe>1)+3, 4, 4), 2, 1)
        cframe /= 2
        
        #input is (ngf*4) x nframe x 32 x 32
        self.conv4 = nn.Conv3d(ngf*4, ngf*8, ((cframe>1)+3, 4, 4), 2, 1)
        cframe /= 2
        
        #input is (ngf*8) x nframe x 16 x 16
        self.conv5 = nn.Conv3d(ngf*8, ngf*8, ((cframe>1)+3, 4, 4), 2, 1)
        cframe /= 2
        
        #input is (ngf*8) x nframe x 8 x 8
        self.conv6 = nn.Conv3d(ngf*8, ngf*8, ((cframe>1)+3, 4, 4), 2, 1)
        cframe /= 2
        
        #input is (ngf*8) x nframe x 4 x 4
        self.conv7 = nn.Conv3d(ngf*8, ngf*8, ((cframe>1)+3, 4, 4), 2, 1)
        cframe /= 2
        
        #input is (ngf*8) x nframe x 2 x 2
        self.conv8 = nn.Conv3d(ngf*8, ngf*8, ((cframe>1)+3, 4, 4), 2, 1)
        cframe /= 2
        
        #input is (ngf*8) x nframe x 1 x 1
        
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)
        
        self.dconv1 = nn.ConvTranspose3d(ngf*8, ngf*8, ((cframe>=1)+3, 4, 4), 2, 1)
        cframe *= 2
        
        #input is (ngf*8) x nframe x 2 x 2
        self.dconv2 = nn.ConvTranspose3d(ngf*8*2, ngf*8, ((cframe>=1)+3, 4, 4), 2, 1)
        cframe *= 2
        
        #input is (ngf*8) x nframe x 4 x 4
        self.dconv3 = nn.ConvTranspose3d(ngf*8*2, ngf*8, ((cframe>=1)+3, 4, 4), 2, 1)
        cframe *= 2
        
        #input is (ngf*8) x nframe x 8 x 8
        self.dconv4 = nn.ConvTranspose3d(ngf*8*2, ngf*8, ((cframe>=1)+3, 4, 4), 2, 1)
        cframe *= 2
        
        #input is (ngf*8) x nframe x 16 x 16
        self.dconv5 = nn.ConvTranspose3d(ngf*8*2, ngf*4, ((cframe>=1)+3, 4, 4), 2, 1)
        cframe *= 2
        
        #input is (ngf*4) x nframe x 32 x 32
        self.dconv6 = nn.ConvTranspose3d(ngf*4*2, ngf*2, ((cframe>=1)+3, 4, 4), 2, 1)
        cframe *= 2
        
        #input is (ngf*2) x nframe x 64 x 64
        self.dconv7 = nn.ConvTranspose3d(ngf*2*2, ngf, ((cframe>=1)+3, 4, 4), 2, 1)
        cframe *= 2
        
        #input is (ngf) x nframe x 128 x 128
        self.dconv8 = nn.ConvTranspose3d(ngf*2, outc, ((cframe>=1)+3, 4, 4), 2, 1)
        
        self.tanh = nn.Tanh()
    
    def forward(self, input, condition):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        inputs = torch.cat([input, condition], 1)
        
        #input is (inc) x 256 x 256
        e1 = nn.parallel.data_parallel(self.conv1, inputs, gpu_ids)
        
        #input is (ngf) x 128 x 128
        e2 = nn.parallel.data_parallel(self.leakyrelu, e1, gpu_ids)
        e2 = nn.parallel.data_parallel(self.conv2, e2, gpu_ids)
        e2 = nn.parallel.data_parallel(self.bn2, e2, gpu_ids)
        
        #input is (ngf*2) x 64 x 64
        e3 = nn.parallel.data_parallel(self.leakyrelu, e2, gpu_ids)
        e3 = nn.parallel.data_parallel(self.conv3, e3, gpu_ids)
        e3 = nn.parallel.data_parallel(self.bn4, e3, gpu_ids)
        
        #input is (ngf*4) x 32 x 32
        e4 = nn.parallel.data_parallel(self.leakyrelu, e3, gpu_ids)
        e4 = nn.parallel.data_parallel(self.conv4, e4, gpu_ids)
        e4 = nn.parallel.data_parallel(self.bn8, e4, gpu_ids)
        
        #input is (ngf*8) x 16 x 16
        e5 = nn.parallel.data_parallel(self.leakyrelu, e4, gpu_ids)
        e5 = nn.parallel.data_parallel(self.conv5, e5, gpu_ids)
        e5 = nn.parallel.data_parallel(self.bn8, e5, gpu_ids)
        
        #input is (ngf*8) x 8 x 8
        e6 = nn.parallel.data_parallel(self.leakyrelu, e5, gpu_ids)
        e6 = nn.parallel.data_parallel(self.conv6, e6, gpu_ids)
        e6 = nn.parallel.data_parallel(self.bn8, e6, gpu_ids)
        
        #input is (ngf*8) x 4 x 4
        e7 = nn.parallel.data_parallel(self.leakyrelu, e6, gpu_ids)
        e7 = nn.parallel.data_parallel(self.conv7, e7, gpu_ids)
        e7 = nn.parallel.data_parallel(self.bn8, e7, gpu_ids)
        
        #input is (ngf*8) x 2 x 2
        e8 = nn.parallel.data_parallel(self.leakyrelu, e7, gpu_ids)
        e8 = nn.parallel.data_parallel(self.conv8, e8, gpu_ids)
        
        #input is (ngf*8) x 1 x 1
        d1 = nn.parallel.data_parallel(self.relu, e8, gpu_ids)
        d1 = nn.parallel.data_parallel(self.dconv1, d1, gpu_ids)
        d1 = nn.parallel.data_parallel(self.bn8, d1, gpu_ids)
        d1 = nn.parallel.data_parallel(self.dropout, d1, gpu_ids)
        
        #input is (ngf*8) x 2 x 2
        d1 = torch.cat([d1, e7], 1)
        d2 = nn.parallel.data_parallel(self.relu, d1, gpu_ids)
        d2 = nn.parallel.data_parallel(self.dconv2, d2, gpu_ids)
        d2 = nn.parallel.data_parallel(self.bn8, d2, gpu_ids)
        d2 = nn.parallel.data_parallel(self.dropout, d2, gpu_ids)
        
        #input is (ngf*8) x 4 x 4
        d2 = torch.cat([d2, e6], 1)
        d3 = nn.parallel.data_parallel(self.relu, d2, gpu_ids)
        d3 = nn.parallel.data_parallel(self.dconv3, d3, gpu_ids)
        d3 = nn.parallel.data_parallel(self.bn8, d3, gpu_ids)
        d3 = nn.parallel.data_parallel(self.dropout, d3, gpu_ids)
        
        #input is (ngf*8) x 8 x 8
        d3 = torch.cat([d3, e5], 1)
        d4 = nn.parallel.data_parallel(self.relu, d3, gpu_ids)
        d4 = nn.parallel.data_parallel(self.dconv4, d4, gpu_ids)
        d4 = nn.parallel.data_parallel(self.bn8, d4, gpu_ids)
        
        #input is (ngf*8) x 16 x 16
        d4 = torch.cat([d4, e4], 1)
        d5 = nn.parallel.data_parallel(self.relu, d4, gpu_ids)
        d5 = nn.parallel.data_parallel(self.dconv5, d5, gpu_ids)
        d5 = nn.parallel.data_parallel(self.bn4, d5, gpu_ids)
        
        #input is (ngf*4) x 32 x 32
        d5 = torch.cat([d5, e3], 1)
        d6 = nn.parallel.data_parallel(self.relu, d5, gpu_ids)
        d6 = nn.parallel.data_parallel(self.dconv6, d6, gpu_ids)
        d6 = nn.parallel.data_parallel(self.bn2, d6, gpu_ids)
        
        #input is (ngf*2) x 64 x 64
        d6 = torch.cat([d6, e2], 1)
        d7 = nn.parallel.data_parallel(self.relu, d6, gpu_ids)
        d7 = nn.parallel.data_parallel(self.dconv7, d7, gpu_ids)
        d7 = nn.parallel.data_parallel(self.bn1, d7, gpu_ids)
        
        #input is (ngf) x 128 x 128
        d7 = torch.cat([d7, e1], 1)
        d8 = nn.parallel.data_parallel(self.relu, d7, gpu_ids)
        d8 = nn.parallel.data_parallel(self.dconv8, d8, gpu_ids)
        
        o1 = nn.parallel.data_parallel(self.tanh, d8, gpu_ids)
        
        return o1
        
        
class DC_3DGAN_G_unet_64(nn.Module):
    def __init__(self, isize, nframe, inc, outc, ngf, ngpu, n_extra_layers=0):
        super(DC_3DGAN_G_unet_64, self).__init__()
        self.ngpu = ngpu
        
        cframe = nframe
        assert isize % 16 == 0, "isize has to be a multiple of 16" 
        
        #input is (inc) x nframe x 64 x 64
        self.conv1 = nn.Conv3d(inc, ngf, 4, 2, 1)
        
        self.leakyrelu = nn.LeakyReLU(True)
        self.bn1 = nn.BatchNorm3d(ngf)
        self.bn2 = nn.BatchNorm3d(ngf*2)
        self.bn4 = nn.BatchNorm3d(ngf*4)
        
        cframe /= 2
        
        #input is (ngf) x nframe x 32 x 32
        self.conv2 = nn.Conv3d(ngf, ngf*2, ((cframe>1)+3, 4, 4), 2, 1)
        cframe /= 2
        
        #input is (ngf*2) x nframe x 16 x 16
        self.conv3 = nn.Conv3d(ngf*2, ngf*4, ((cframe>1)+3, 4, 4), 2, 1)
        cframe /= 2
        
        #input is (ngf*4) x nframe x 8 x 8
        self.conv4 = nn.Conv3d(ngf*4, ngf*4, ((cframe>1)+3, 4, 4), 2, 1)
        cframe /= 2
        
        #input is (ngf*4) x nframe x 4 x 4
        self.conv5 = nn.Conv3d(ngf*4, ngf*4, ((cframe>1)+3, 4, 4), 2, 1)
        cframe /= 2
        
        #input is (ngf*4) x nframe x 2 x 2
        self.conv6 = nn.Conv3d(ngf*4, ngf*4, ((cframe>1)+3, 4, 4), 2, 1)
        cframe /= 2
        
        #input is (ngf*4) x nframe x 1 x 1
        
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)
        
        self.dconv1 = nn.ConvTranspose3d(ngf*4, ngf*4, ((cframe>=1)+3, 4, 4), 2, 1)
        cframe *= 2
        
        #input is (ngf*4) x nframe x 2 x 2
        self.dconv2 = nn.ConvTranspose3d(ngf*4*2, ngf*4, ((cframe>=1)+3, 4, 4), 2, 1)
        cframe *= 2
        
        #input is (ngf*4) x nframe x 4 x 4
        self.dconv3 = nn.ConvTranspose3d(ngf*4*2, ngf*4, ((cframe>=1)+3, 4, 4), 2, 1)
        cframe *= 2
        
        #input is (ngf*4) x nframe x 8 x 8
        self.dconv4 = nn.ConvTranspose3d(ngf*4*2, ngf*2, ((cframe>=1)+3, 4, 4), 2, 1)
        cframe *= 2
        
        #input is (ngf*2) x nframe x 16 x 16
        self.dconv5 = nn.ConvTranspose3d(ngf*2*2, ngf, ((cframe>=1)+3, 4, 4), 2, 1)
        cframe *= 2
        
        #input is (ngf) x nframe x 32 x 32
        self.dconv6 = nn.ConvTranspose3d(ngf*2, outc, ((cframe>=1)+3, 4, 4), 2, 1)
        
        self.tanh = nn.Tanh()
    
    def forward(self, input, condition):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        inputs = torch.cat([input, condition], 1)
        
        #input is (inc) x 64 x 64
        e1 = nn.parallel.data_parallel(self.conv1, inputs, gpu_ids)
        
        #input is (ngf) x 32 x 32
        e2 = nn.parallel.data_parallel(self.leakyrelu, e1, gpu_ids)
        e2 = nn.parallel.data_parallel(self.conv2, e2, gpu_ids)
        e2 = nn.parallel.data_parallel(self.bn2, e2, gpu_ids)
        
        #input is (ngf*2) x 16 x 16
        e3 = nn.parallel.data_parallel(self.leakyrelu, e2, gpu_ids)
        e3 = nn.parallel.data_parallel(self.conv3, e3, gpu_ids)
        e3 = nn.parallel.data_parallel(self.bn4, e3, gpu_ids)
        
        #input is (ngf*4) x 8 x 8
        e4 = nn.parallel.data_parallel(self.leakyrelu, e3, gpu_ids)
        e4 = nn.parallel.data_parallel(self.conv4, e4, gpu_ids)
        e4 = nn.parallel.data_parallel(self.bn4, e4, gpu_ids)
        
        #input is (ngf*4) x 4 x 4
        e5 = nn.parallel.data_parallel(self.leakyrelu, e4, gpu_ids)
        e5 = nn.parallel.data_parallel(self.conv5, e5, gpu_ids)
        e5 = nn.parallel.data_parallel(self.bn4, e5, gpu_ids)
        
        #input is (ngf*4) x 2 x 2
        e6 = nn.parallel.data_parallel(self.leakyrelu, e5, gpu_ids)
        e6 = nn.parallel.data_parallel(self.conv6, e6, gpu_ids)
        
        
        #input is (ngf*4) x 1 x 1
        d1 = nn.parallel.data_parallel(self.relu, e6, gpu_ids)
        d1 = nn.parallel.data_parallel(self.dconv1, d1, gpu_ids)
        d1 = nn.parallel.data_parallel(self.bn4, d1, gpu_ids)
        d1 = nn.parallel.data_parallel(self.dropout, d1, gpu_ids)
        
        #input is (ngf*4) x 2 x 2
        d1 = torch.cat([d1, e5], 1)
        d2 = nn.parallel.data_parallel(self.relu, d1, gpu_ids)
        d2 = nn.parallel.data_parallel(self.dconv2, d2, gpu_ids)
        d2 = nn.parallel.data_parallel(self.bn4, d2, gpu_ids)
        d2 = nn.parallel.data_parallel(self.dropout, d2, gpu_ids)
        
        #input is (ngf*4) x 4 x 4
        d2 = torch.cat([d2, e4], 1)
        d3 = nn.parallel.data_parallel(self.relu, d2, gpu_ids)
        d3 = nn.parallel.data_parallel(self.dconv3, d3, gpu_ids)
        d3 = nn.parallel.data_parallel(self.bn4, d3, gpu_ids)
        d3 = nn.parallel.data_parallel(self.dropout, d3, gpu_ids)
        
        #input is (ngf*4) x 8 x 8
        d3 = torch.cat([d3, e3], 1)
        d4 = nn.parallel.data_parallel(self.relu, d3, gpu_ids)
        d4 = nn.parallel.data_parallel(self.dconv4, d4, gpu_ids)
        d4 = nn.parallel.data_parallel(self.bn2, d4, gpu_ids)
        
        #input is (ngf*2) x 16 x 16
        d4 = torch.cat([d4, e2], 1)
        d5 = nn.parallel.data_parallel(self.relu, d4, gpu_ids)
        d5 = nn.parallel.data_parallel(self.dconv5, d5, gpu_ids)
        d5 = nn.parallel.data_parallel(self.bn1, d5, gpu_ids)
        
        #input is (ngf) x 32 x 32
        d5 = torch.cat([d5, e1], 1)
        d6 = nn.parallel.data_parallel(self.relu, d5, gpu_ids)
        d6 = nn.parallel.data_parallel(self.dconv6, d6, gpu_ids)
        
        o1 = nn.parallel.data_parallel(self.tanh, d6, gpu_ids)
        
        return o1
        
class DC_3DGAN_G_unet_128(nn.Module):
    def __init__(self, isize, nframe, inc, outc, ngf, ngpu, n_extra_layers=0):
        super(DC_3DGAN_G_unet_128, self).__init__()
        self.ngpu = ngpu
        
        cframe = nframe
        assert isize % 16 == 0, "isize has to be a multiple of 16" 
        
        #input is (inc) x nframe x 128 x 128
        self.conv1 = nn.Conv3d(inc, ngf, 4, 2, 1)
        
        self.leakyrelu = nn.LeakyReLU(True)
        self.bn1 = nn.BatchNorm3d(ngf)
        self.bn2 = nn.BatchNorm3d(ngf*2)
        self.bn4 = nn.BatchNorm3d(ngf*4)
        self.bn8 = nn.BatchNorm3d(ngf*8)
        
        cframe /= 2
        
        #input is (ngf) x nframe x 64 x 64
        self.conv2 = nn.Conv3d(ngf, ngf*2, ((cframe>1)+3, 4, 4), 2, 1)
        cframe /= 2
        
        #input is (ngf*2) x nframe x 32 x 32
        self.conv3 = nn.Conv3d(ngf*2, ngf*4, ((cframe>1)+3, 4, 4), 2, 1)
        cframe /= 2
        
        #input is (ngf*4) x nframe x 16 x 16
        self.conv4 = nn.Conv3d(ngf*4, ngf*8, ((cframe>1)+3, 4, 4), 2, 1)
        cframe /= 2
        
        #input is (ngf*8) x nframe x 8 x 8
        self.conv5 = nn.Conv3d(ngf*8, ngf*8, ((cframe>1)+3, 4, 4), 2, 1)
        cframe /= 2
        
        #input is (ngf*8) x nframe x 4 x 4
        self.conv6 = nn.Conv3d(ngf*8, ngf*8, ((cframe>1)+3, 4, 4), 2, 1)
        cframe /= 2
        
        #input is (ngf*8) x nframe x 2 x 2
        self.conv7 = nn.Conv3d(ngf*8, ngf*8, ((cframe>1)+3, 4, 4), 2, 1)
        cframe /= 2
        
        #input is (ngf*8) x nframe x 1 x 1
        
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)
        
        self.dconv1 = nn.ConvTranspose3d(ngf*8, ngf*8, ((cframe>=1)+3, 4, 4), 2, 1)
        cframe *= 2
        
        #input is (ngf*8) x nframe x 2 x 2
        self.dconv2 = nn.ConvTranspose3d(ngf*8*2, ngf*8, ((cframe>=1)+3, 4, 4), 2, 1)
        cframe *= 2
        
        #input is (ngf*8) x nframe x 4 x 4
        self.dconv3 = nn.ConvTranspose3d(ngf*8*2, ngf*8, ((cframe>=1)+3, 4, 4), 2, 1)
        cframe *= 2
        
        #input is (ngf*8) x nframe x 8 x 8
        self.dconv4 = nn.ConvTranspose3d(ngf*8*2, ngf*4, ((cframe>=1)+3, 4, 4), 2, 1)
        cframe *= 2
        
        #input is (ngf*4) x nframe x 16 x 16
        self.dconv5 = nn.ConvTranspose3d(ngf*4*2, ngf*2, ((cframe>=1)+3, 4, 4), 2, 1)
        cframe *= 2
        
        #input is (ngf*2) x nframe x 32 x 32
        self.dconv6 = nn.ConvTranspose3d(ngf*2*2, ngf, ((cframe>=1)+3, 4, 4), 2, 1)
        cframe *= 2
        
        #input is (ngf) x nframe x 64 x 64
        self.dconv7 = nn.ConvTranspose3d(ngf*2, outc, ((cframe>=1)+3, 4, 4), 2, 1)
        
        self.tanh = nn.Tanh()
    
    def forward(self, input, condition):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        inputs = torch.cat([input, condition], 1)
        
        #input is (inc) x 128 x 128
        e1 = nn.parallel.data_parallel(self.conv1, inputs, gpu_ids)
        
        #input is (ngf) x 64 x 64
        e2 = nn.parallel.data_parallel(self.leakyrelu, e1, gpu_ids)
        e2 = nn.parallel.data_parallel(self.conv2, e2, gpu_ids)
        e2 = nn.parallel.data_parallel(self.bn2, e2, gpu_ids)
        
        #input is (ngf*2) x 32 x 32
        e3 = nn.parallel.data_parallel(self.leakyrelu, e2, gpu_ids)
        e3 = nn.parallel.data_parallel(self.conv3, e3, gpu_ids)
        e3 = nn.parallel.data_parallel(self.bn4, e3, gpu_ids)
        
        #input is (ngf*4) x 16 x 16
        e4 = nn.parallel.data_parallel(self.leakyrelu, e3, gpu_ids)
        e4 = nn.parallel.data_parallel(self.conv4, e4, gpu_ids)
        e4 = nn.parallel.data_parallel(self.bn8, e4, gpu_ids)
        
        #input is (ngf*8) x 8 x 8
        e5 = nn.parallel.data_parallel(self.leakyrelu, e4, gpu_ids)
        e5 = nn.parallel.data_parallel(self.conv5, e5, gpu_ids)
        e5 = nn.parallel.data_parallel(self.bn8, e5, gpu_ids)
        
        #input is (ngf*8) x 4 x 4
        e6 = nn.parallel.data_parallel(self.leakyrelu, e5, gpu_ids)
        e6 = nn.parallel.data_parallel(self.conv6, e6, gpu_ids)
        e6 = nn.parallel.data_parallel(self.bn8, e6, gpu_ids)
        
        #input is (ngf*8) x 2 x 2
        e7 = nn.parallel.data_parallel(self.leakyrelu, e6, gpu_ids)
        e7 = nn.parallel.data_parallel(self.conv7, e7, gpu_ids)
        
        
        #input is (ngf*8) x 1 x 1
        d1 = nn.parallel.data_parallel(self.relu, e7, gpu_ids)
        d1 = nn.parallel.data_parallel(self.dconv1, d1, gpu_ids)
        d1 = nn.parallel.data_parallel(self.bn8, d1, gpu_ids)
        d1 = nn.parallel.data_parallel(self.dropout, d1, gpu_ids)
        
        #input is (ngf*8) x 2 x 2
        d1 = torch.cat([d1, e6], 1)
        d2 = nn.parallel.data_parallel(self.relu, d1, gpu_ids)
        d2 = nn.parallel.data_parallel(self.dconv2, d2, gpu_ids)
        d2 = nn.parallel.data_parallel(self.bn8, d2, gpu_ids)
        d2 = nn.parallel.data_parallel(self.dropout, d2, gpu_ids)
        
        #input is (ngf*8) x 4 x 4
        d2 = torch.cat([d2, e5], 1)
        d3 = nn.parallel.data_parallel(self.relu, d2, gpu_ids)
        d3 = nn.parallel.data_parallel(self.dconv3, d3, gpu_ids)
        d3 = nn.parallel.data_parallel(self.bn8, d3, gpu_ids)
        d3 = nn.parallel.data_parallel(self.dropout, d3, gpu_ids)
        
        #input is (ngf*8) x 8 x 8
        d3 = torch.cat([d3, e4], 1)
        d4 = nn.parallel.data_parallel(self.relu, d3, gpu_ids)
        d4 = nn.parallel.data_parallel(self.dconv4, d4, gpu_ids)
        d4 = nn.parallel.data_parallel(self.bn4, d4, gpu_ids)
        
        #input is (ngf*4) x 16 x 16
        d4 = torch.cat([d4, e3], 1)
        d5 = nn.parallel.data_parallel(self.relu, d4, gpu_ids)
        d5 = nn.parallel.data_parallel(self.dconv5, d5, gpu_ids)
        d5 = nn.parallel.data_parallel(self.bn2, d5, gpu_ids)
        
        #input is (ngf*2) x 32 x 32
        d5 = torch.cat([d5, e2], 1)
        d6 = nn.parallel.data_parallel(self.relu, d5, gpu_ids)
        d6 = nn.parallel.data_parallel(self.dconv6, d6, gpu_ids)
        d6 = nn.parallel.data_parallel(self.bn1, d6, gpu_ids)
        
        #input is (ngf) x 64 x 64
        d6 = torch.cat([d6, e1], 1)
        d7 = nn.parallel.data_parallel(self.relu, d6, gpu_ids)
        d7 = nn.parallel.data_parallel(self.dconv7, d7, gpu_ids)
        
        o1 = nn.parallel.data_parallel(self.tanh, d7, gpu_ids)
        
        return o1