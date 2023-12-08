# author: Daniel-Ji (e-mail: gepengai.ji@gmail.com)
# data: 2021-01-16
# torch libraries
import os
import logging
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
from lib.DGNet import DGNet as Network
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim
from torchvision.utils import make_grid
# customized libraries
import eval.python.metrics as Measure
from utils.utils import clip_gradient
from utils.dataset import get_loader, test_dataset
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
    
    
class decoder(nn.Module):
    def __init__(self, num_layers=8, num_features=32, out_num=1):
        super(decoder, self).__init__()
        
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
        #output = output.reshape(output.size(0), -1)
        
        
        
        
        return output
    
    
def norm(tensor_map):
    tensor_map_AA=tensor_map.clone()
    tensor_map_AA = tensor_map_AA.view(tensor_map.size(0), -1)
    tensor_map_AA -= tensor_map_AA.min(1, keepdim=True)[0]
    tensor_map_AA /= (tensor_map_AA.max(1, keepdim=True)[0]-tensor_map_AA.min(1, keepdim=True)[0])
    tensor_map_AA = tensor_map_AA.view(tensor_map.shape)
    #tensor_map_AA[torch.isnan(tensor_map_AA)]=0
    
    return tensor_map_AA

def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, model,encoder_model, decoder_model, optimizer, optimizer_3, optimizer_4,cos, cos1, l2,save_dir, sig, epoch, save_path, writer):
    
    
    """
    train function
    """
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    
    state_sig = {
         'state_dict_encoder':encoder_model.state_dict(),
        'optimizer_3': optimizer_3.state_dict(),
          'state_dict_decoder':decoder_model.state_dict(),
        'optimizer_4': optimizer_4.state_dict(),
          'state_dict_model':model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    
    try:
        for i, (images, gts, grads) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            optimizer.zero_grad()
            optimizer_3.zero_grad()
            optimizer_4.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            grads = grads.cuda()
            signal=encoder_model(images)
            preds = model(images*signal)
            signal_rec=decoder_model(images*signal)
            signal_AA=norm(signal)
            signal_rec_AA=norm(signal_rec)

            loss1=(1. - cos(signal_AA.reshape( signal_AA.size(0), -1), gts[:,0,:].reshape( gts[:,0,:].size(0), -1)))
            loss_tot1=torch.sum(loss1)

            loss2=(1. - cos(signal_rec_AA.reshape( signal_rec_AA.size(0), -1), gts[:,0,:].reshape( gts[:,0,:].size(0), -1)))
            loss_tot2=torch.sum(loss2)
            loss_pred = structure_loss(preds[0], gts)
            loss_grad = grad_loss_func(preds[1], grads)
            
            loss = 0.1*loss_pred + 0.1*loss_grad  + 10*loss_tot1 + 10*loss_tot2

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            optimizer_3.step()
            optimizer_4.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} loss_pred: {:.4f} loss_grad: {:0.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss_pred.data, loss_grad.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} loss_pred: {:.4f} loss_grad: {:0.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss.data, loss_pred.data, loss_grad.data))
                # TensorboardX-Loss
                writer.add_scalars('Loss_Statistics',
                                   {'loss_pred': loss_pred.data, 'loss_grad': loss_grad.data,
                                    'Loss_total': loss.data},
                                   global_step=step)
                # TensorboardX-Training Data
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('GT', grid_image, step)

                # TensorboardX-Outputs
                res = preds[0][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_final', torch.tensor(res), step, dataformats='HW')
                res = preds[1][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_grad', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        #if epoch % 1 == 0:
        torch.save(state_sig, os.path.join("%s/result_9/%s"% (save_dir, sig), 'state_{}.pickle'.format( epoch+1)))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(state_sig, os.path.join("%s/result_9/%s"% (save_dir, sig), 'state_{}.pickle'.format( epoch+1)))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer):
    """
    validation function
    """
    global best_metric_dict, best_score, best_epoch
    FM = Measure.Fmeasure()
    SM = Measure.Smeasure()
    EM = Measure.Emeasure()
    metrics_dict = dict()

    model.eval()
    with torch.no_grad():
        for i in range(test_loader.size):
            image, gt, _, _ = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            image = image.cuda()

            res = model(image)

            res = F.upsample(res[0], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            FM.step(pred=res, gt=gt)
            SM.step(pred=res, gt=gt)
            EM.step(pred=res, gt=gt)

        metrics_dict.update(Sm=SM.get_results()['sm'])
        metrics_dict.update(mxFm=FM.get_results()['fm']['curve'].max().round(3))
        metrics_dict.update(mxEm=EM.get_results()['em']['curve'].max().round(3))

        cur_score = metrics_dict['Sm'] + metrics_dict['mxFm'] + metrics_dict['mxEm']

        if epoch == 1:
            best_score = cur_score
            print('[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(
                epoch, metrics_dict['mxFm'], metrics_dict['Sm'], metrics_dict['mxEm']))
            logging.info('[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(
                epoch, metrics_dict['mxFm'], metrics_dict['Sm'], metrics_dict['mxEm']))
        else:
            if cur_score > best_score:
                best_metric_dict = metrics_dict
                best_score = cur_score
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('>>> save state_dict successfully! best epoch is {}.'.format(epoch))
            else:
                print('>>> not find the best epoch -> continue training ...')
            print('[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})\n[Best Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(
                epoch, metrics_dict['mxFm'], metrics_dict['Sm'], metrics_dict['mxEm'],
                best_epoch, best_metric_dict['mxFm'], best_metric_dict['Sm'], best_metric_dict['mxEm']))
            logging.info('[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})\n[Best Epoch:{}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(
                epoch, metrics_dict['mxFm'], metrics_dict['Sm'], metrics_dict['mxEm'],
                best_epoch, best_metric_dict['mxFm'], best_metric_dict['Sm'], best_metric_dict['mxEm']))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=24, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--model', type=str, default='DGNet', 
                        choices=['DGNet', 'DGNet-S', 'DGNet-PVTv2-B0', 'DGNet-PVTv2-B1', 'DGNet-PVTv2-B2', 'DGNet-PVTv2-B3', 'DGNet-PVTv2-B4'])
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--train_root', type=str, default='../dataset/TrainDataset/',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='../dataset/TestDataset/CAMO/',
                        help='the test rgb images root')
    parser.add_argument('--gpu_id', type=str, default='1', 
                        help='train use gpu')
    parser.add_argument('--save_path', type=str, default='./lib_pytorch/snapshot/Exp02/',
                        help='the path to save model and log')
    opt = parser.parse_args()

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

    # build the model
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

    grad_loss_func = torch.nn.MSELoss()

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Imgs/',
                              gt_root=opt.train_root + 'GT/',
                              grad_root=opt.train_root + 'Gradient-Foreground/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=4)
    val_loader = test_dataset(image_root=opt.val_root + 'Imgs/',
                              gt_root=opt.val_root + 'GT/',
                              testsize=opt.trainsize)
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info(">>> current mode: network-train/val")
    logging.info('>>> config: {}'.format(opt))
    print('>>> config: : {}'.format(opt))

    step = 0
    writer = SummaryWriter(save_path + 'summary')

    best_score = 0
    best_epoch = 0

    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-5)
    print(">>> start train...")
    
    encoder_model=encoder()
    encoder_model.cuda()

    optimizer_3 = torch.optim.Adam(encoder_model.parameters(), lr=0.00001)
    #self.decoder_model=nn.DataParallel(decoder()) 
    decoder_model=decoder()

    decoder_model.cuda()
    optimizer_4 = torch.optim.Adam(decoder_model.parameters(), lr=0.00001)
    
    

    sig = str(datetime.now())

    print(sig)    

    save_dir="/mnt/scratch/asnanivi/runs"
    os.makedirs('%s/logs/%s' % (save_dir, sig), exist_ok=True)
    os.makedirs('%s/result_9/%s' % (save_dir, sig), exist_ok=True)


    cos = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
    cos1 = nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    l2=torch.nn.MSELoss().cuda()
    
    
    for epoch in range(1, opt.epoch):
        # schedule
        cosine_schedule.step()
        writer.add_scalar('learning_rate', cosine_schedule.get_lr()[0], global_step=epoch)
        logging.info('>>> current lr: {}'.format(cosine_schedule.get_lr()[0]))
        # train
        train(train_loader, model,encoder_model, decoder_model, optimizer, optimizer_3, optimizer_4,cos, cos1, l2,save_dir, sig, epoch, save_path, writer)
        
        if epoch > opt.epoch//2:
            # validation
            val(val_loader, model, epoch, save_path, writer)
