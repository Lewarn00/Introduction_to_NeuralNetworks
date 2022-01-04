import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required


def get_optimiser(models, mode, args):
    params_models = []
    reduced_params = []

    removed_params = []

    skip_lists = ['bn', 'bias']

    for m in models:

        m_skip = []
        m_noskip = []

        params_models += list(m.parameters())

        for name, param in m.named_parameters():
            if (any(skip_name in name for skip_name in skip_lists)):
                m_skip.append(param)
            else:
                m_noskip.append(param)
        reduced_params += list(m_noskip)
        removed_params += list(m_skip)
    if mode == 'pretrain':
        lr = args.learning_rate
        wd = args.weight_decay
    else:
        lr = args.finetune_learning_rate
        wd = args.finetune_weight_decay

    if args.optimiser == 'adam':

        optimiser = optim.Adam(params_models, lr=lr,
                               weight_decay=wd)

    elif args.optimiser == 'sgd':

        optimiser = optim.SGD(params_models, lr=lr,
                              weight_decay=wd, momentum=0.9)

    elif args.optimiser == 'lars':

        print("reduced_params len: {}".format(len(reduced_params)))
        print("removed_params len: {}".format(len(removed_params)))

        optimiser = LARS(reduced_params+removed_params, lr=lr,
                         weight_decay=wd, eta=0.001, use_nesterov=False, len_reduced=len(reduced_params))
    else:

        raise NotImplementedError('{} not setup.'.format(args.optimiser))

    return optimiser
