from __future__ import print_function
import time
import socket
import argparse

import os
import math
import random
import numpy as np
from glob import glob
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader

# ---load model arc---
from model_archs.TTST_arc import TTST as net

from torch.utils.tensorboard import SummaryWriter

def PSNR(pred, gt):
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    logprint(net)
    logprint('Total number of parameters: %f M' % (num_params / 1e6))


def checkpoint(epoch):
    model_out_path = opt.save_folder + "ttst_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    logprint("Checkpoint saved to {}".format(model_out_path))


rewrite_print = print
def print_log(*arg):
    file_path = opt.save_folder + '/train_log.txt'
    rewrite_print(*arg)
    rewrite_print(*arg, file=open(file_path, "a", encoding="utf-8"))


def normalize(img, entire_mean: list, entire_std: list, inverse: bool = False):
    """Using given mean and std to normalize images.
    If inverse is True, do the inverse process.

    Args:
        images: NCHW or CHW
    Return:
        images
    """
    images = img.clone()
    def normalize_standard(image, mean, std):
        if isinstance(image, torch.Tensor):
            return torch.divide(
                torch.add(image, -torch.tensor(mean)),
                torch.maximum(torch.tensor(std), torch.tensor(1e-5)),
            )
        else:
            if not isinstance(image, np.ndarray):
                image = np.asarray(image)
            return (image - mean) / max(std, 1e-5)

    def inverse_normalize_standard(image, mean, std):
        if isinstance(image, torch.Tensor):
            return torch.add(
                torch.multiply(
                    image, torch.maximum(torch.tensor(std), torch.tensor(1e-5))
                ),
                torch.tensor(mean),
            )
        else:
            if not isinstance(image, np.ndarray):
                image = np.asarray(image)
            return image * max(std, 1e-5) + mean

    if images.dim() == 3:
        c, _, _ = images.shape
        for j in range(c):
            if inverse:
                images[j] = inverse_normalize_standard(
                    images[j], entire_mean[j], entire_std[j]
                )
            else:
                images[j] = normalize_standard(images[j], entire_mean[j], entire_std[j])
    elif images.dim() == 4:
        n, c, _, _ = images.shape
        for y in range(n):
            for j in range(c):
                if inverse:
                    images[y][j] = inverse_normalize_standard(
                        images[y][j], entire_mean[j], entire_std[j]
                    )
                else:
                    images[y][j] = normalize_standard(
                        images[y][j], entire_mean[j], entire_std[j]
                    )
    return images


def upscale(feat, scale_factor: int = 2):
    # resolution decrease
    if scale_factor == 1:
        return feat
    else:
        return F.avg_pool2d(feat, scale_factor)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


transform2tensor = T.ToTensor()
transform2Pil = T.ToPILImage()
def augment(img_in, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    for c in range(2):
        lr = transform2Pil(img_in[c])
        hr = transform2Pil(img_tar[c])

        if random.random() < 0.5 and flip_h:
            lr = ImageOps.flip(lr)
            hr = ImageOps.flip(hr)
            #img_bic = ImageOps.flip(img_bic)
            info_aug['flip_h'] = True

        if rot:
            if random.random() < 0.5:
                lr = ImageOps.mirror(lr)
                hr = ImageOps.mirror(hr)
                #img_bic = ImageOps.mirror(img_bic)
                info_aug['flip_v'] = True
            if random.random() < 0.5:
                lr = lr.rotate(180)
                hr = hr.rotate(180)
                #img_bic = img_bic.rotate(180)
                info_aug['trans'] = True
        img_in[c] = transform2tensor(lr)
        img_tar[c] = transform2tensor(hr)
    return img_in, img_tar, info_aug


class get_dataset(Dataset):
    def __init__(self, data_augmentation, scale_factor=4, image_type='Wind', dataset_device = 'PC', is_train = True):
        """
        Args:
            image_type: 'Solar', 'Wind'
        """
        if image_type == 'Solar':
            years = ['07', '08', '09', '10', '11', '12', '13']
            if dataset_device == 'HPCC':
                path_train = [
                    '/lustre/scratch/guiyli/Dataset_NSRDB/npyFiles/dni_dhi/20' + i
                    for i in years
                ]
                path_test = '/lustre/scratch/guiyli/Dataset_NSRDB/npyFiles/dni_dhi/2014'
            elif dataset_device == 'PC':
                path_train = None
                path_test = '/home/guiyli/Documents/DataSet/Solar/npyFiles/dni_dhi/2014'
                print('No training data set on current device')
            entire_mean = '392.8659294288083,125.10559238383577'
            entire_std = '351.102247720423,101.6698946847449'
        elif image_type == 'Wind':
            years = ['07', '08', '09', '10', '11', '12', '13']
            if dataset_device == 'HPCC':
                path_train = [
                    '/lustre/scratch/guiyli/Dataset_WIND/npyFiles/20' + i + '/u_v' for i in years
                ]
                path_test = '/lustre/scratch/guiyli/Dataset_WIND/npyFiles/2014/u_v'
            elif dataset_device == 'PC':
                path_train = [
                    '/home/guiyli/Documents/DataSet/Wind/20' + i + '/u_v' for i in years
                ]
                path_test = '/home/guiyli/Documents/DataSet/Wind/2014/u_v'
            entire_mean = '-0.6741845839785552,-1.073033474161022'
            entire_std = '5.720375778518578,4.772050058088903'

        self.entire_mean = [float(i) for i in entire_mean.split(',')]
        self.entire_std = [float(i) for i in entire_std.split(',')]

        self.root = path_train if is_train else path_test

        self.files = []
        if type(self.root) == list:
            for i in self.root:
                self.files += glob(i + '/*.npy')
            assert self.files is not None, 'No data found.'
            self.suffix = 'npy'
        else:
            self.files = glob(self.root + '/*.tif')
            self.suffix = 'tif'
            if not self.files:  # in case images are in npy format
                self.files = glob(self.root + '/*.npy')
                assert self.files is not None, 'No data found.'
                self.suffix = 'npy'

        self.resize2tensor = transforms.Compose(
            [
                transforms.ToTensor(),  # convert from HWC to CHW
                transforms.Resize(
                    (8*scale_factor, 8*scale_factor),
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
            ]
        )

        self.scale_factor = scale_factor
        self.toTensor = transforms.ToTensor()
        self.data_augmentation = data_augmentation
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if isinstance(index, str):
            f = glob(self.root + '/' + index + '.' + self.suffix)[0]
        else:
            f = self.files[index]

        if self.suffix == 'tif':
            img = Image.open(f).astype(np.float32)
        else:
            img = np.load(f).astype(np.float32)

        img = self.resize2tensor(img)
        img = normalize(img, self.entire_mean, self.entire_std)

        lr = upscale(img, self.scale_factor)

        if self.data_augmentation:
            lr, img, _ = augment(lr, img)
        return lr, img


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.0001')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')

parser.add_argument('--data_type', type=str, default='Wind')
parser.add_argument('--data_device', type=str, default='PC')
parser.add_argument('--verbose', type=boolean_string, default=True)

parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--residual', type=bool, default=False, help='Use global resudial or not')
parser.add_argument('--pretrained_sr', default='saved_models/ttst/xx.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='saved_models/ttst/', help='Location to save checkpoint models')
parser.add_argument('--log_folder', default='tb_logs/ttst/', help='Location to save checkpoint models')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
cuda = opt.gpu_mode
verbose = opt.verbose
def logprint(*args):
    if verbose:
        print(*args)

logprint(opt)

# -------- save training log ---------------
current_time = time.strftime("%H-%M-%S")
opt.save_folder = opt.save_folder + current_time + '/'
opt.log_folder = opt.log_folder + current_time + '/'
writer = SummaryWriter('./{}'.format(opt.log_folder))

os.makedirs(opt.log_folder, exist_ok=True)
os.makedirs(opt.save_folder, exist_ok=True)


torch.cuda.manual_seed(opt.seed)

logprint('===> Loading training datasets')
train_set = get_dataset(opt.data_augmentation, opt.upscale_factor, opt.data_type, opt.data_device, True)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
logprint('===> Loading val datasets')
val_set = get_dataset(opt.data_augmentation, opt.upscale_factor, opt.data_type, opt.data_device, False)
val_data_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=1, shuffle=False)

logprint('===> Building model ttst')
model = net(in_chans=2, img_size=8, upscale=opt.upscale_factor)
model = torch.nn.DataParallel(model, device_ids=gpus_list)
logprint('---------- Networks architecture -------------')
print_network(model)
model = model.cuda(gpus_list[0])

if opt.pretrained:
    model_name = os.path.join(opt.pretrained_sr)
    logprint('load model', model_name)
    if os.path.exists(model_name):
        # model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        logprint('Pre-trained SR model is loaded.')
L1_criterion = nn.L1Loss()
L1_criterion = L1_criterion.cuda(gpus_list[0])
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

best_epoch = 0
best_test_psnr = 0.0
for epoch in range(opt.start_epoch, opt.nEpochs + 1):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        lr, gt = batch[0], batch[1]
        if cuda:
            gt = Variable(gt).cuda(gpus_list[0])
            lr = Variable(lr).cuda(gpus_list[0])

        optimizer.zero_grad()

        t0 = time.time()
        prediction = model(lr)
        t1 = time.time()

        loss = L1_criterion(prediction, gt)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        logprint("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader),
                                                                                 loss.item(), (t1 - t0)))
    print_log("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    writer.add_scalar('Avg. Loss', epoch_loss / len(training_data_loader), epoch)

    # val while training
    count = 1
    avg_psnr_predicted = 0.0
    avg_test_psnr = 0.0
    model.eval()
    for batch in val_data_loader:
        lr, gt = batch[0], batch[1]
        with torch.no_grad():
            gt = Variable(gt).cuda(gpus_list[0])
            lr = Variable(lr).cuda(gpus_list[0])
        with torch.no_grad():
            t0 = time.time()
            prediction = model(lr)
            t1 = time.time()

        prediction = prediction.cpu()
        prediction = prediction.data[0].numpy().astype(np.float32)
        prediction = prediction * 255.0
        prediction = prediction.clip(0, 255)

        gt = gt.cpu()
        gt = gt.squeeze().numpy().astype(np.float32)
        gt = gt * 255.
        psnr_predicted = PSNR(prediction, gt)
        print_log("===> Processing image: %s || Timer: %.4f sec. || PSNR: %.4f dB" % (str(count), (t1 - t0), psnr_predicted))
        avg_psnr_predicted += psnr_predicted
        avg_test_psnr = avg_psnr_predicted / len(val_data_loader)
        count += 1
    if avg_test_psnr > best_test_psnr:
        best_epoch = epoch
        best_test_psnr = avg_test_psnr
    print_log("===> Epoch {} Complete: Avg. PSNR: {:.4f} Best Epoch {} Best PSNR: {:.4f}".format(epoch,
                                                                                                 avg_psnr_predicted / len(val_data_loader),
                                                                                                 best_epoch, best_test_psnr))
    writer.add_scalar('Avg. PSNR', avg_psnr_predicted / len(val_data_loader), epoch)

    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch + 1) % (opt.nEpochs / 2) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 2.0
        logprint('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    if epoch % (opt.snapshots) == 0:
        checkpoint(epoch)
