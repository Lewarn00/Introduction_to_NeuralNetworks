import torch
import torch.nn as nn
import torch.nn.functional as F

import model.network as models


class MoCo_Model(nn.Module):
    def __init__(self, args, queue_size=65536, momentum=0.999, temperature=0.07):

        super(MoCo_Model, self).__init__()

        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        assert self.queue_size % args.batch_size == 0  

        self.encoder_q = getattr(models, args.model)(
            args, num_classes=128) 

        self.encoder_k = getattr(models, args.model)(
            args, num_classes=128)  

        self.encoder_q.fc = models.projection_MLP(args)
        self.encoder_k.fc = models.projection_MLP(args)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(self.queue_size, 128))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update(self):
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.momentum + p_q.detach().data * (1. - self.momentum)

    @torch.no_grad()
    def shuffled_idx(self, batch_size):
        shuffled_idxs = torch.randperm(batch_size).long().cuda()

        reverse_idxs = torch.zeros(batch_size).long().cuda()

        value = torch.arange(batch_size).long().cuda()

        reverse_idxs.index_copy_(0, shuffled_idxs, value)

        return shuffled_idxs, reverse_idxs

    @torch.no_grad()
    def update_queue(self, feat_k):
        batch_size = feat_k.size(0)

        ptr = int(self.queue_ptr)

        self.queue[ptr:ptr + batch_size, :] = feat_k

        ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr

    def InfoNCE_logits(self, f_q, f_k):
        f_k = f_k.detach()

        f_mem = self.queue.clone().detach()

        f_q = nn.functional.normalize(f_q, dim=1)
        f_k = nn.functional.normalize(f_k, dim=1)
        f_mem = nn.functional.normalize(f_mem, dim=1)

        pos = torch.bmm(f_q.view(f_q.size(0), 1, -1),
                        f_k.view(f_k.size(0), -1, 1)).squeeze(-1)

        neg = torch.mm(f_q, f_mem.transpose(1, 0))

        logits = torch.cat((pos, neg), dim=1)

        logits /= self.temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        return logits, labels

    def forward(self, x_q, x_k):

        batch_size = x_q.size(0)

        feat_q = self.encoder_q(x_q)

        shuffled_idxs, reverse_idxs = self.shuffled_idx(batch_size)

        with torch.no_grad():

            self.momentum_update()
            x_k = x_k[shuffled_idxs]
            feat_k = self.encoder_k(x_k)
            feat_k = feat_k[reverse_idxs]

        logit, label = self.InfoNCE_logits(feat_q, feat_k)

        self.update_queue(feat_k)

        return logit, label
