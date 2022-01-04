import os
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10

from utils import *


def get_dataloaders(args):
    dataset = 'CIFAR10'

    args.class_names = (
        'plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
    ) 

    args.crop_dim = 32
    args.n_channels, args.n_classes = 3, 10
    working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', dataset)

    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    dataset_paths = {'train': os.path.join(working_dir, 'train'),
                    'test':  os.path.join(working_dir, 'test')}
    dataloaders = cifar_dataloader(args, dataset_paths)

    return dataloaders, args

def cifar_dataloader(args, dataset_paths):
    color_jitter = transforms.ColorJitter(
        0.8*args.jitter_d, 0.8*args.jitter_d, 0.8*args.jitter_d, 0.2*args.jitter_d)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=args.jitter_p)

    rnd_grey = transforms.RandomGrayscale(p=args.grey_p)

    transf = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            rnd_color_jitter,
            rnd_grey,
            transforms.RandomResizedCrop((args.crop_dim, args.crop_dim), scale=(0.25, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                 (0.24703223, 0.24348513, 0.26158784))]),
        'pretrain': transforms.Compose([
            transforms.ToPILImage(),
            rnd_color_jitter,
            rnd_grey,
            transforms.RandomResizedCrop((args.crop_dim, args.crop_dim)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                 (0.24703223, 0.24348513, 0.26158784))]),
        'test':  transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                 (0.24703223, 0.24348513, 0.26158784))]),
    }

    config = {'train': True, 'test': False}

    datasets = {i: CIFAR10(root=dataset_paths[i], transform=transf[i],
                           train=config[i], download=True) for i in config.keys()}
    val_samples = 500



    f_s_weights = sample_weights(datasets['train'].targets)
    data, labels = random_split(data=datasets['train'].data,
                                labels=datasets['train'].targets,
                                n_classes=args.n_classes,
                                n_samples_per_class=np.repeat(val_samples, args.n_classes).reshape(-1))


    datasets['train_valid'] = datasets['train']
    datasets['pretrain'] = CustomDataset(data=data['train'],
                                         labels=labels['train'], transform=transf['pretrain'], two_crop=args.twocrop)

    datasets['train'] = CustomDataset(data=data['train'],
                                      labels=labels['train'], transform=transf['train'], two_crop=False)

    datasets['valid'] = CustomDataset(data=data['valid'],
                                      labels=labels['valid'], transform=transf['test'], two_crop=False)

    s_weights = sample_weights(datasets['pretrain'].labels)
    config = {
        'pretrain': WeightedRandomSampler(s_weights,
                                          num_samples=len(s_weights), replacement=True),
        'train': WeightedRandomSampler(s_weights,
                                       num_samples=len(s_weights), replacement=True),
        'train_valid': WeightedRandomSampler(f_s_weights,
                                             num_samples=len(f_s_weights), replacement=True),
        'valid': None, 'test': None
    }

    dataloaders = {i: DataLoader(datasets[i], sampler=config[i],
                                 num_workers=8, pin_memory=True, drop_last=True,
                                 batch_size=args.batch_size) for i in config.keys()}

    return dataloaders
