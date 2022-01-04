To run training (this runs training, then fine-tuning): python3 main.py 
To run only fine-tuning: python3 main.py --finetune --load_checkpoint_dir /path/to/checkpoint.pt

Sources:

K. He, et. al Momentum Contrast for Unsupervised Visual Representation Learning

X. Chen, et. al Improved Baselines with Momentum Contrastive Learning

facebookresearch MoCo Code

HobbitLong CMC

T. Chen, et. al SimCLR Paper

noahgolmant pytorch-lars

fabio-deep Distributed-Pytorch-Boilerplate

AidenDurrant Unofficial Pytorch Implementation of MocCoV2