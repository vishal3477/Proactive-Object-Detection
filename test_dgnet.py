import os
import torch
import argparse
import numpy as np
from scipy import misc

import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision.models.utils import load_state_dict_from_url

from utils.dataset import test_dataset as EvalDataset
from lib.DGNet import DGNet as Network
from datetime import datetime
import torch.nn as nn

class encoder(nn.Module):
    def __init__(self, num_layers=8, num_features=32, out_num=1):
        super(encoder, self).__init__()
        
        layers_0 = [nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        
        layers_1=[]
        layers_2=[]
        for i in range(15):
            layers_1.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                      nn.ReLU(inplace=True)))
        
        
            
        self.layers_0 = nn.Sequential(*layers_0)
        self.layers_1 = nn.Sequential(*layers_1)
        
        self.layers_2=nn.Sequential(nn.Conv2d(num_features, 1, kernel_size=3, padding=1),
                                        nn.ReLU(inplace=True))
        #self.layers_2[0].bias=torch.nn.Parameter(torch.ones((1)))
        #self.layers_2[0].weight.data *= 0.0001 
    def forward(self, inputs):
        output = self.layers_0(inputs)
        output = self.layers_1(output)
        output = self.layers_2(output)
        
        
        
        return output
    
    

    
def norm(tensor_map):
    tensor_map_AA=tensor_map.clone()
    tensor_map_AA = tensor_map_AA.view(tensor_map.size(0), -1)
    tensor_map_AA -= tensor_map_AA.min(1, keepdim=True)[0]
    tensor_map_AA /= (tensor_map_AA.max(1, keepdim=True)[0]-tensor_map_AA.min(1, keepdim=True)[0])
    tensor_map_AA = tensor_map_AA.view(tensor_map.shape)
    #tensor_map_AA[torch.isnan(tensor_map_AA)]=0
    
    return tensor_map_AA

def evaluator(model, val_root, map_save_path, trainsize=352):
    val_loader = EvalDataset(image_root=val_root + 'Imgs/',
                             gt_root=val_root + 'GT/',
                             testsize=trainsize)
    encoder_model=encoder()
    encoder_model.cuda()

    optimizer_3 = torch.optim.Adam(encoder_model.parameters(), lr=0.00001)
    
    
    
    
    load_name = opt.model_path
    checkpoint = torch.load(load_name)
    model.load_state_dict(checkpoint['state_dict_model'])
    encoder_model.load_state_dict(checkpoint['state_dict_encoder'])
    
    

    model.eval()
    with torch.no_grad():
        for i in range(val_loader.size):
            image, gt, name, _ = val_loader.load_data()
            gt = np.asarray(gt, np.float32)

            image = image.cuda()
            signal=encoder_model(image)
            output = model(image*signal)
            signal_rec=decoder_model(image*signal)
            #output = model(image)
            output = F.upsample(output[0], size=gt.shape, mode='bilinear', align_corners=False)
            output = output.sigmoid().data.cpu().numpy().squeeze()
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)

            misc.imsave(map_save_path + name, output)
            print('>>> saving prediction at: {}'.format(map_save_path + name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='DGNet', 
                        choices=['DGNet', 'DGNet-S', 'DGNet-PVTv2-B0', 'DGNet-PVTv2-B1', 'DGNet-PVTv2-B2', 'DGNet-PVTv2-B3', 'DGNet-PVTv2-B4'])
    parser.add_argument('--snap_path', type=str, default='./snapshot/DGNet/Net_epoch_best.pth',
                        help='train use gpu')
    parser.add_argument('--gpu_id', type=str, default='1',
                        help='train use gpu')
    parser.add_argument('--model_path', dest='model_path',
                      help='wrapper model path'
                      , type=str)
    
    opt = parser.parse_args()

    txt_save_path = './result/proactive/{}/'.format(opt.snap_path.split('/')[-2])
    os.makedirs(txt_save_path, exist_ok=True)

    print('>>> configs:', opt)

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    elif opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    elif opt.gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')

    cudnn.benchmark = True
    if opt.model == 'DGNet':
        model = Network(channel=64, arc='EfficientNet-B4', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'DGNet-S':
        model = Network(channel=32, arc='EfficientNet-B1', M=[8, 8, 8], N=[8, 16, 32]).cuda()
    elif opt.model == 'DGNet-PVTv2-B0':
        model = Network(channel=32, arc='PVTv2-B0', M=[8, 8, 8], N=[8, 16, 32]).cuda()
    elif opt.model == 'DGNet-PVTv2-B1':
        model = Network(channel=64, arc='PVTv2-B1', M=[8, 8, 8], N=[4, 8, 16]).cuda()   
    elif opt.model == 'DGNet-PVTv2-B2':
        model = Network(channel=64, arc='PVTv2-B2', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'DGNet-PVTv2-B3':
        model = Network(channel=64, arc='PVTv2-B3', M=[8, 8, 8], N=[4, 8, 16]).cuda()   
    elif opt.model == 'DGNet-PVTv2-B4':
        model = Network(channel=64, arc='PVTv2-B4', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    else:
        raise Exception("Invalid Model Symbol: {}".format(opt.model))
    
    # TODO: remove FC layers from snapshots
    #model.load_state_dict(torch.load(opt.snap_path), strict=False)
    
    
    
    
    
    
    model.eval()

    #for data_name in ['CAMO', 'COD10K', 'NC4K']:
    for data_name in ['NC4K']:
        map_save_path = txt_save_path + "{}/".format(data_name)
        os.makedirs(map_save_path, exist_ok=True)
        evaluator(
            model=model,
            val_root='/mnt/ufs18/nodr/home/asnanivi/cod/cod_2/TestDataset/' + data_name + '/',
            map_save_path=map_save_path,
            trainsize=352)
