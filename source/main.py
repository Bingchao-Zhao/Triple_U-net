import torch
import metrics 
import data
import model
import os
import argparse
import skimage
import numpy as np
from matplotlib import pyplot as plt
import time
from utils import *
import Config
import cv2
import transform
from skimage import measure
import skimage.io as io
def _create_optimizer(conf, model):
    optimizer_config    = conf.optim_conf
    learning_rate       = optimizer_config['learning_rate']
    weight_decay        = optimizer_config['weight_decay']
    betas               =  optimizer_config['betas']
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,\
                                         model.parameters()),betas=betas,\
                                         lr=learning_rate, weight_decay=weight_decay)#
    return optimizer

def _create_lr_scheduler(conf, optimizer):
    lr_scheduler = conf.lr_scheduler
    gamma = lr_scheduler['gamma']
    return torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                        gamma=gamma, last_epoch=-1)

def save_plot(input,name,tile='LOSS_plot',ylabel='loss',xlabel='free'):
    #['acc', 'loss', 'val_acc', 'val_loss']
    for i in range(len(input)-1):
        plt.plot(input[i][0],input[i][1])
    plt.title(tile)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(input[len(input)-1], loc='upper left')
    plt.savefig(name)
    my_print('Successfully save plot:{}'.format(name))

def evaluation(pre,mask,cutoff,file,min_size=10):
    IOU=np.array([])
    DICE=np.array([])
    AJI=np.array([])
    TP=np.array([])
    PQ = np.array([])
    aji=0.
    f1=0.

    for i in range(len(pre)):
            img=skimage.morphology.remove_small_objects(np.array(pre[i])>cutoff, min_size=min_size)
            PQ          =   np.append(PQ,metrics.get_fast_pq(np.array(mask[i],dtype='uint8'),np.array(img,dtype='uint8'))[0][2])
            IOU         =   np.append(IOU,metrics.compute_iou(img,mask[i],cutoff))
            DICE        =   np.append(DICE,metrics.compute_F1(img,mask[i],cutoff))
            AJI         =   np.append(AJI,metrics.get_fast_aji(mask[i],img))
            TP          =   np.append(TP,metrics.compute_TP_ratio(img,mask[i],cutoff))


    my_print('Num is:{} '.format(len(PQ)),'cutoff=[{}]'.format(cutoff),'PQ=[{:.6}]'.format(np.mean(PQ)),
                'DICE=[{:.6}]'.format(np.mean(DICE)),
                'AJI=[{:.6}]'.format(np.mean(AJI)))

    return  np.mean(PQ),np.mean(DICE),np.mean(AJI)   

class test_model(object):
    def __init__(self,conf):
        super(test_model, self).__init__()
        self.conf =conf
        self.net = torch.load(conf.model_path)
        self.datagen=data.trainGenerator(conf.test_data_path,conf.label_path,conf.edg_path)

    def test(self):
        rgb_pre=[]
        file_ = []
        HE_pre = []
        nuclei_pre=[]
        grounf_truth=[]
        file_ = []
        epoch_loss=.0
        for index in range(len(self.datagen)):
            rtime_print('{}/{}'.format(index+1,len(self.datagen)))
            img,file = self.datagen.load_image(index)
            if self.datagen.just_img_name(file):
                continue
            HE  = self.datagen.load_HE(index)
            img = torch.unsqueeze(img.cuda(), 0)
            HE  = torch.unsqueeze(HE.cuda(), 0)
            
            mask        = self.datagen.load_mask(index)
            nuclei,outh,outrgb =self.predition(img,HE,file)
            rgb_pre.append(torch.squeeze(outrgb.cpu()))
            nuclei_pre.append(nuclei)
            HE_pre.append(torch.squeeze(outh.cpu()))
            grounf_truth.append(torch.squeeze(mask))
            file_.append(file)

        ret=evaluation(nuclei_pre,grounf_truth,self.conf.cutoff,file_)
    
    def predition(self,img,HE,file):
        with torch.no_grad(): 
            nuclei,H,RGB = self.net(img,HE)
            nuclei=torch.squeeze(nuclei.cpu())
            RGB=torch.squeeze(RGB.cpu())
            H=torch.squeeze(H.cpu())
            io.imsave(os.path.join(self.conf.save_path,file[0:(len(file)-4)]+'-pre.png'),np.array((nuclei>self.conf.cutoff)*255,dtype='uint8'))
            io.imsave(os.path.join(self.conf.save_path,file[0:(len(file)-4)]+'-RGB.png'),np.array(RGB*255,dtype='uint8'))                      
            io.imsave(os.path.join(self.conf.save_path,file[0:(len(file)-4)]+'-H.png'),np.array(H*255,dtype='uint8'))   

        return nuclei,H,RGB

class train(object):
    def __init__(self,conf):
        super(train, self).__init__()
        self.conf = conf
        self.net       = model.net().cuda()
        self.data_Generator=data.trainGenerator
        
        self.optimizer = _create_optimizer(conf, self.net)
        self.scheduler = _create_lr_scheduler(conf, self.optimizer)
        
        data_transform =  transform.Compose([transform.RandomMirror_h(),
            transform.RandomMirror_w(),
            transform.rotation(),
            transform.flip(),
            transform.elastic_transform()])

        self.train_data_Generator= self.data_Generator(self.conf.train_data_path,
                                        self.conf.label_path,self.conf.edg_path,data_transform)
        self.train_loader = torch.utils.data.DataLoader(
               self.train_data_Generator,
               batch_size=self.conf.batch_size,
               pin_memory=True,
               shuffle=True,
               collate_fn=data.collater)
                          
    def epoch_train(self,epoch):
        Loss=.0
        i=0
        for d in self.train_loader:
            self.net.train()
            img,H,ground_truth,edge=d
            H =H.cuda()
            img=img.cuda()
            i=i+img.size()[0]
            rtime_print('{}/{}'.format(i,len(self.train_data_Generator)))
            with torch.enable_grad():   
                self.optimizer.zero_grad()
                nuclei,contourH,contourRGB = self.net(img,H)
                loss_nuclei = data.soft_dice(nuclei, ground_truth)

                loss_nucleice = data.soft_truncate_ce_loss(nuclei, ground_truth)
                loss_H=data.soft_dice(contourH, edge)
                loss_RGB=data.seg_loss(contourRGB, ground_truth)

                loss = 0.3*loss_nuclei+loss_nucleice+0.3*loss_H+0.3*loss_RGB
                loss.backward()
                self.optimizer.step()
                Loss = Loss+loss.item()
        return Loss/len(self.train_data_Generator)
        
    def training(self):
        loss_record=[]
        epoch_record=[]

        for epoch in range(self.conf.epoches):
            my_print('Epoch {}/{}'.format(epoch,self.conf.epoches - 1),'-' * 60)
            start = time.time()
            Loss=self.epoch_train(epoch)
            self.scheduler.step()
            end = time.time()
            my_print('Epoch{} Loss:{:.6}  cost time:{:.6}'.\
                            format(epoch,Loss,str(end-start)))

            if epoch>5 and epoch%5==0:
                torch.save(self.net,'model/model-{}.hdf5'.format(epoch))#
                rtime_print('Save AJI model',end='\n')

            loss_record.append(Loss)
            epoch_record.append(epoch+1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train model')
    parser.add_argument('--epoch',default=100,required=False,type=int,
					help='num of epoch')
    parser.add_argument('--train',default=1,required=False,type=int,
					help='1:trian or 0:test')
    args            = parser.parse_args()
     
    conf = Config.config()
    conf.epoches = args.epoch
    conf.train = args.train
    start = time.time()
    end = 0
    if conf.train:
        my_print('data_path:        {}'.format(conf.train_data_path))
        my_print('label_path:       {}'.format(conf.label_path))
        my_print('valid_data_path:  {}'.format(conf.valid_data_path))
        my_print('epoches num:      {}'.format(conf.epoches))
        
        my_activation = train(conf)
        my_print('Total epoches ({})'.format(conf.epoches))
        my_activation.training()
        end = time.time()
    else:
        my_print('test_data_path: {}'.format(conf.test_data_path))
        my_activation = test_model(conf)
        my_activation.test()
        end = time.time()
    my_print('Running time:{}'.format(str(end-start)))


