import sys, os
import cv2
import scipy.misc
import numpy as np
from random import shuffle

#def dataloader(dataset, dataroot, batchSize, frameSize, cropSize, scale=255.0):
#    if dataset == 'UCF101':

            
class dataloader():
    def __init__(self, dataset, dataroot, dataSize, batchSize, frameSize, cropSize, scale=255.0):
        self.checkpoint = 0
        self.dataset = dataset
        self.dataroot = dataroot
        self.dataSize = dataSize
        self.batchSize = batchSize
        self.frameSize = frameSize
        self.cropSize = cropSize
        self.scale = scale
        self.files = os.listdir(dataroot)
        
    def reset(self):
        self.checkpoint = 0
        shuffle(self.files)
        
    def load(self):
        if self.dataset == 'UCF101':
            if self.checkpoint == len(self.files):
                return True
            data = []#np.zeros([self.dataSize, self.batchSize, self.frameSize, self.cropSize, self.cropSize, 3], dtype='float32')
            dataiter = 0
            batchiter = 0
            batch = np.zeros([self.batchSize, 3, self.frameSize, self.cropSize, self.cropSize], dtype='float32')
            for i, file in enumerate(self.files[self.checkpoint:]):
                self.checkpoint += 1
                sample = np.zeros([self.frameSize, 3, self.cropSize, self.cropSize], dtype='float32')
                frameiter = 0
                
                cap = cv2.VideoCapture(self.dataroot+'/'+file)
                
                ret = True
                while ret:
                    ret, frame = cap.read()
                    if ret == False:
                        continue
                    h, w, d = frame.shape
                    if h < w:
                        frame = scipy.misc.imresize(frame, (self.cropSize, int(w/h*self.cropSize)))
                        startw = frame.shape[1]//2-self.cropSize//2
                        frame = frame[:, startw:(startw+self.cropSize), :]
                    else:
                        frame = scipy.misc.imresize(frame, (int(h/w*self.cropSize), self.cropSize))
                        starth = frame.shape[0]//2-self.cropSize//2
                        frame = frame[starth:(starth+self.cropSize), :, :]
                    
                    frame_scale = np.array(frame/self.scale, dtype='float32')
                    frame_scale = cv2.cvtColor(frame_scale, cv2.COLOR_BGR2YUV)
                    frame_scale = (frame_scale-0.5)/0.5
                    
                    sample[frameiter, :, :, :] = np.swapaxes(np.swapaxes(frame_scale, 0, 1), 0, 2)
                    
                    frameiter += 1
                    if frameiter == self.frameSize:
                        batch[batchiter] = np.swapaxes(sample, 0, 1)
                        frameiter = 0
                        batchiter += 1
                        if batchiter == self.batchSize:
                            data.append(batch)
                            dataiter += 1
                            batchiter = 0
                            #print ('dataloader: '+ str(dataiter))
                            if len(data) == self.dataSize:
                                data = np.array(data, dtype='float32')
                                data = data.reshape(self.batchSize*self.dataSize, 3, self.frameSize, self.cropSize, self.cropSize)
                                tmp = np.arange(data.shape[0])
                                shuffle(tmp)
                                return data[tmp].reshape(self.dataSize, self.batchSize, 3, self.frameSize, self.cropSize, self.cropSize)
                                
                cap.release()
                
            data = np.array(data, dtype='float32')
            l = data.shape[0]
            data = data.reshape(-1, 3, self.frameSize, self.cropSize, self.cropSize)
            tmp = np.arange(data.shape[0])
            shuffle(tmp)
            return data[tmp].reshape(l, self.batchSize, 3, self.frameSize, self.cropSize, self.cropSize)
            
    def save(self, num, data, prefix):
        for i in range(num):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(prefix+str(i)+'.avi', fourcc, 30.0, (self.cropSize, self.cropSize))
            sample = data[i]
            
            sample = np.swapaxes(sample, 0, 1)
            sample = np.swapaxes(sample, 1, 2)
            sample = np.swapaxes(sample, 2, 3)
            
            sample = (sample+1.0)/2.0
            
            for j in range(sample.shape[0]):
                bgr = cv2.cvtColor(sample[j], cv2.COLOR_YUV2BGR)
                bgr_scale = np.array(bgr*self.scale, dtype='uint8')
                out.write(bgr_scale)
                
            out.release()
        