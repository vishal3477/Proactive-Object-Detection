# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import datetime
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import edge_detection


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
class encoder(nn.Module):
    def __init__(self, num_layers=8, num_features=32, out_num=1):
        super(encoder, self).__init__()
        
        layers_0 = [nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        
        layers_1=[]
        layers_2=[]
        for i in range(5):
            layers_1.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                      nn.ReLU(inplace=True)))
        
        
            
        self.layers_0 = nn.Sequential(*layers_0)
        self.layers_1 = nn.Sequential(*layers_1)
        
        self.layers_2=nn.Sequential(nn.Conv2d(num_features, 1, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(1),
                                        nn.ReLU(inplace=True))
        
        
    def forward(self, inputs):
        output = self.layers_0(inputs)
        output = self.layers_1(output)
        output = self.layers_2(output)
        
        
        
        return output
    
    
class decoder(nn.Module):
    def __init__(self, num_layers=8, num_features=32, out_num=1):
        super(decoder, self).__init__()
        
        layers_0 = [nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        
        layers_1=[]
        layers_2=[]
        for i in range(3):
            layers_1.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                      nn.ReLU(inplace=True)))
        
        
            
        self.layers_0 = nn.Sequential(*layers_0)
        self.layers_1 = nn.Sequential(*layers_1)
        
        self.layers_2=nn.Sequential(nn.Conv2d(num_features, 1, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(1),
                                        nn.ReLU(inplace=True))
        
        
        
        
        self.fc1 = nn.Linear(4, 256)
        
        self.fc2 = nn.Linear(256, 16384)
        
        self.upsample = torch.nn.Upsample(scale_factor=16, mode='nearest')
        
        
        
        
    def forward(self, inputs, boxes):
        output = self.layers_0(inputs)
        output = self.layers_1(output)
        output = self.layers_2(output)
        #output = output.reshape(output.size(0), -1)
        
        feats_boxes=self.fc1(boxes)
        feats_boxes=self.fc2(feats_boxes)
        feats_boxes = feats_boxes.reshape(feats_boxes.size(0), 1,128,128)
        feats_boxes=self.upsample(feats_boxes)
        out_final=output+ F.interpolate(feats_boxes, size=(output.shape[2], output.shape[3]))
        
        
        return out_final
    
class vector_var(nn.Module):
    def __init__(self , size):
        super(vector_var, self).__init__()
        A = torch.rand(1,size,size, device='cpu')
        self.A = nn.Parameter(A)
        
    def forward(self):
        return self.A
    
def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0,n,None) 
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n,None,None)
                  for i in range(X.dim()))
    #print(axis,n,f_idx,b_idx)
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front],axis)

def fftshift(real, imag):
    for dim in range(1, len(real.size())):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return real, imag

def norm(tensor_map):
    tensor_map_AA=tensor_map.clone()
    tensor_map_AA = tensor_map_AA.view(tensor_map.size(0), -1)
    tensor_map_AA -= tensor_map_AA.min(1, keepdim=True)[0]
    tensor_map_AA /= (tensor_map_AA.max(1, keepdim=True)[0]-tensor_map_AA.min(1, keepdim=True)[0])
    tensor_map_AA = tensor_map_AA.view(tensor_map.shape)
    #tensor_map_AA[torch.isnan(tensor_map_AA)]=0
    
    return tensor_map_AA

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=50, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and diaplay
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')

  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train"#+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "vg":
      # train sizes: train, smalltrain, minitrain
      # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)
  sig = str(datetime.datetime.now())

  print(sig)    
    
  
  save_dir="/mnt/scratch/asnanivi/runs"
  os.makedirs('%s/logs/%s' % (save_dir, sig), exist_ok=True)
  os.makedirs('%s/result_9/%s' % (save_dir, sig), exist_ok=True)



  encoder_model=encoder().cuda()
  optimizer_encoder_model = torch.optim.Adam(encoder_model.parameters(), lr=0.0001)
    
    
  encoder_model= nn.DataParallel(encoder_model)
  
    
    
  decoder_model=decoder().cuda() 
  optimizer_decoder_model = torch.optim.Adam(decoder_model.parameters(), lr=0.0001)
  decoder_model= nn.DataParallel(decoder_model)
  
    
    
  state_sig = {
     'state_dict_encoder':encoder_model.state_dict(),
    'optimizer_encoder_model': optimizer_encoder_model.state_dict(),
      'state_dict_decoder':decoder_model.state_dict(),
    'optimizer_decoder_model': optimizer_decoder_model.state_dict()
}
  
  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
  train_size = len(roidb)

  print('{:d} roidb entries'.format(len(roidb)))

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  sampler_batch = sampler(train_size, args.batch_size)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_data1 = torch.FloatTensor(1)

  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_data1 = im_data1.cuda()
    
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True
  print("classes", len(imdb.classes))
  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.cuda:
    fasterRCNN.cuda()
      
  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
  
  
  load_name = './faster_rcnn_1_6_14657.pth'
  print("loading checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  args.session = checkpoint['session']
  args.start_epoch = checkpoint['epoch']
  fasterRCNN.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  lr = optimizer.param_groups[0]['lr']
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']
  
  print("loaded pretrained checkpoint %s" % (load_name))
  
  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  iters_per_epoch = int(train_size / args.batch_size)

  
  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")

    
  l2=torch.nn.MSELoss().cuda()
  cos = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
  cos1 = nn.CosineSimilarity(dim=0, eps=1e-6).cuda()

  high_filter=edge_detection.Net(args.batch_size)
  high_filter=high_filter.cuda()

  for epoch in range(args.start_epoch, args.max_epochs + 1):
    # setting to train mode
    fasterRCNN.train()
    loss_temp = 0
    start = time.time()

    if epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    data_iter = iter(dataloader)
    for step in range(iters_per_epoch):
      data = next(data_iter)
      with torch.no_grad():
           im_data1.resize_(data[0].size()).copy_(data[0])
           im_info.resize_(data[1].size()).copy_(data[1])
           gt_boxes.resize_(data[2].size()).copy_(data[2])
           num_boxes.resize_(data[3].size()).copy_(data[3])
      print(im_info.shape)
      print(gt_boxes.shape)
      print(num_boxes.shape)
      signal=encoder_model(im_data1)
      
      
        
      im_data=im_data1.clone()*signal
      fasterRCNN.zero_grad()
      print(im_data.shape)
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
      
      signal_rec=decoder_model(im_data, torch.mean(gt_boxes[:,:,0:4], dim=1))
      
      signal_AA=norm(signal)
      signal_rec_AA=norm(signal_rec)
        
      
      signal_att=torch.zeros((im_data.shape[0],1,im_data.shape[2],im_data.shape[3]), dtype=torch.float32).cuda()
  
      for batch_num in range(im_data.shape[0]):
        for i_num in range(50):
          signal_att[batch_num,:,int(gt_boxes[batch_num,i_num, 0]):int(gt_boxes[batch_num,i_num, 0])+int(gt_boxes[batch_num,i_num, 2]),int(gt_boxes[batch_num,i_num, 1]):int(gt_boxes[batch_num,i_num, 1])+int(gt_boxes[batch_num,i_num, 3])]=1
    
      loss_rcnn = 10*(rpn_loss_cls.mean() + rpn_loss_box.mean() \
           + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean())
      
      #zero=torch.zeros(signal_AA.shape, dtype=torch.float32).cuda()
      edges_gt=high_filter(im_data)
      
      loss1=(1. - cos(signal_AA.reshape( signal_AA.size(0), -1), signal_att[:,0,:].reshape( signal_att[:,0,:].size(0), -1)))
      loss_tot1=10*torch.sum(loss1)
    
      loss2=(1. - cos(signal_rec_AA.reshape( signal_rec_AA.size(0), -1), signal_att[:,0,:].reshape( signal_att[:,0,:].size(0), -1)))
      loss_tot2=10*torch.sum(loss2)
        
      
      
      loss=7*loss_rcnn + 10*loss_tot1 + 10*loss_tot2 
      
      print(loss_tot1, loss_tot2, loss3)
    
    
      # backward
      optimizer.zero_grad()
      
      optimizer_encoder_model.zero_grad()
      optimizer_decoder_model.zero_grad()
      
      loss.backward()
      if args.net == "vgg16":
          clip_gradient(fasterRCNN, 10.)
      optimizer.step()
      
      optimizer_encoder_model.step()
      optimizer_decoder_model.step()
    
      
    

      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= (args.disp_interval + 1)

        if args.mGPUs:
          loss_rpn_cls = rpn_loss_cls.mean().item()
          loss_rpn_box = rpn_loss_box.mean().item()
          loss_rcnn_cls = RCNN_loss_cls.mean().item()
          loss_rcnn_box = RCNN_loss_bbox.mean().item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
        else:
          loss_rpn_cls = rpn_loss_cls.item()
          loss_rpn_box = rpn_loss_box.item()
          loss_rcnn_cls = RCNN_loss_cls.item()
          loss_rcnn_box = RCNN_loss_bbox.item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch, loss_rcnn, lr))
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
        print("loss1=%f,loss2=%f"%( loss_tot1, loss_tot2))
        if args.use_tfboard:
          info = {
            'loss': loss_temp,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_box': loss_rpn_box,
            'loss_rcnn_cls': loss_rcnn_cls,
            'loss_rcnn_box': loss_rcnn_box
          }
          logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

        loss_temp = 0
        start = time.time()
        if step%500==0:
          save_name = os.path.join(args.save_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
          save_checkpoint({
           'session': args.session,
           'epoch': epoch + 1,
           'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
           'optimizer': optimizer.state_dict(),
           'pooling_mode': cfg.POOLING_MODE,
           'class_agnostic': args.class_agnostic,
           }, save_name)
          torch.save(state_sig, args.save_dir, 'signal_state__{}_{}_{}.pickle'.format(args.session, epoch, step)))
          print('save model: {}'.format(save_name))

    


  if args.use_tfboard:
    logger.close()
