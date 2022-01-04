import os
import logging
import random
import configargparse
import warnings
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from train import finetune, evaluate, pretrain
from datasets import get_dataloaders
from utils import *
import model.network as models
from model.moco import MoCo_Model


warnings.filterwarnings("ignore")


default_config = os.path.join(os.path.split(os.getcwd())[0], 'config.conf')

parser = configargparse.ArgumentParser(
    description='Pytorch MocoV2', default_config_files=[default_config])
parser.add_argument('-c', '--my-config', required=False,
                    is_config_file=True, help='config file path')
parser.add_argument('--model', default='resnet18',
                    help='Model, (Options: resnet18).')
parser.add_argument('--n_epochs', type=int, default=1000,
                    help='Number of Epochs in Contrastive Training.')
parser.add_argument('--finetune_epochs', type=int, default=100,
                    help='Number of Epochs in Linear Classification Training.')
parser.add_argument('--warmup_epochs', type=int, default=10,
                    help='Number of Warmup Epochs During Contrastive Training.')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of Samples Per Batch.')
parser.add_argument('--learning_rate', type=float, default=1.0,
                    help='Starting Learing Rate for Contrastive Training.')
parser.add_argument('--base_lr', type=float, default=0.0001,
                    help='Base / Minimum Learing Rate to Begin Linear Warmup.')
parser.add_argument('--finetune_learning_rate', type=float, default=0.1,
                    help='Starting Learing Rate for Linear Classification Training.')
parser.add_argument('--weight_decay', type=float, default=1e-6,
                    help='Contrastive Learning Weight Decay Regularisation Factor.')
parser.add_argument('--finetune_weight_decay', type=float, default=0.0,
                    help='Linear Classification Training Weight Decay Regularisation Factor.')
parser.add_argument('--optimiser', default='sgd',
                    help='Optimiser, (Options: sgd, adam).')
parser.add_argument('--patience', default=50, type=int,
                    help='Number of Epochs to Wait for Improvement.')
parser.add_argument('--queue_size', type=int, default=65536,
                    help='Size of Memory Queue, Must be Divisible by batch_size.')
parser.add_argument('--queue_momentum', type=float, default=0.999,
                    help='Momentum for the Key Encoder Update.')
parser.add_argument('--temperature', type=float, default=0.07,
                    help='InfoNCE Temperature Factor')
parser.add_argument('--jitter_d', type=float, default=1.0,
                    help='Distortion Factor for the Random Colour Jitter Augmentation')
parser.add_argument('--jitter_p', type=float, default=0.8,
                    help='Probability to Apply Random Colour Jitter Augmentation')
parser.add_argument('--blur_sigma', nargs=2, type=float, default=[0.1, 2.0],
                    help='Radius to Apply Random Colour Jitter Augmentation')
parser.add_argument('--blur_p', type=float, default=0.5,
                    help='Probability to Apply Gaussian Blur Augmentation')
parser.add_argument('--grey_p', type=float, default=0.2,
                    help='Probability to Apply Random Grey Scale')
parser.add_argument('--no_twocrop', dest='twocrop', action='store_false',
                    help='Whether or Not to Use Two Crop Augmentation, Used to Create Two Views of the Input for Contrastive Learning. (Default: True)')
parser.set_defaults(twocrop=True)
parser.add_argument('--load_checkpoint_dir', default=None,
                    help='Path to Load Pre-trained Model From.')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Perform Only Linear Classification Training. (Default: False)')
parser.set_defaults(finetune=False)


def setup():
    local_rank = None
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seed = 44
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

    return device, local_rank


def main():
    args = parser.parse_args()
    device, local_rank = setup()
    dataloaders, args = get_dataloaders(args)


    args = experiment_config(parser, args)

    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))


    if any(args.model in model_name for model_name in model_names):
        base_encoder = getattr(models, args.model)(
            args, num_classes=args.n_classes)  

    else:
        raise NotImplementedError("Model Not Implemented: {}".format(args.model))

    for name, param in base_encoder.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    init_weights(base_encoder)


    moco = MoCo_Model(args, queue_size=args.queue_size,
                      momentum=args.queue_momentum, temperature=args.temperature)

    if torch.cuda.device_count() > 1:
        moco = nn.DataParallel(moco)
        base_encoder = nn.DataParallel(base_encoder)

    print('\nUsing', torch.cuda.device_count(), 'GPU(s).\n')

    moco.to(device)
    base_encoder.to(device)

    args.print_progress = True

    if args.print_progress:
        print_network(moco, args) 
        logging.info('\npretrain/train: {} - valid: {} - test: {}'.format(
            len(dataloaders['train'].dataset), len(dataloaders['valid'].dataset),
            len(dataloaders['test'].dataset)))

    if not args.finetune:
        pretrain(moco, dataloaders, args)
        base_encoder = load_moco(base_encoder, args)

        finetune(base_encoder, dataloaders, args)
        test_loss, test_acc, test_acc_top5 = evaluate(
            base_encoder, dataloaders, 'test', args.finetune_epochs, args)

        print('[Test] loss {:.4f} - acc {:.4f} - acc_top5 {:.4f}'.format(
            test_loss, test_acc, test_acc_top5))

    else:
        base_encoder = load_moco(base_encoder, args)
        finetune(base_encoder, dataloaders, args)
        test_loss, test_acc, test_acc_top5 = evaluate(
            base_encoder, dataloaders, 'test', args.finetune_epochs, args)

        print('[Test] loss {:.4f} - acc {:.4f} - acc_top5 {:.4f}'.format(
            test_loss, test_acc, test_acc_top5))


if __name__ == '__main__':
    main()
