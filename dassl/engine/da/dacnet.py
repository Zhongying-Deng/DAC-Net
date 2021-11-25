import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Function

from dassl.data import DataManager
from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.data.transforms import build_transform
from dassl.engine.ssl import FixMatch
from dassl.evaluation import build_evaluator
from dassl.engine.trainer import SimpleNet
from dassl.utils import count_num_param
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import load_pretrained_weights


def domain_discrepancy(out1, out2, loss_type):
    def huber_loss(e, d=1):
        t =torch.abs(e)
        ret = torch.where(t < d, 0.5 * t ** 2, d * (t - 0.5 * d))
        return torch.mean(ret)

    diff = out1 - out2
    if loss_type == 'L1':
        loss = torch.mean(torch.abs(diff))
    elif loss_type == 'Huber':
        loss = huber_loss(diff)
    else:
        loss = torch.mean(diff*diff)
    return loss



@TRAINER_REGISTRY.register()
class DACNet(FixMatch):
    """
    Zhongying Deng, Kaiyang Zhou, Yongxin Yang and Tao Xiang. 
    'Domain Attention Consistency for Multi-Source Domain Adaptation'. BMVC 2021
    
    https://www.bmvc2021-virtualconference.com/assets/papers/0353.pdf
    
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.weight_u = cfg.TRAINER.FIXMATCH.WEIGHT_U
        self.conf_thre = cfg.TRAINER.FIXMATCH.CONF_THRE
        self.ema_alpha = cfg.TRAINER.FIXMATCH.EMA_ALPHA
        self.domain_loss_type = cfg.TRAINER.DACNET.LOSS_TYPE
        self.weight_d = cfg.TRAINER.DACNET.WEIGHT_D  # domain attention consistency loss weight
        self.weight_con = cfg.TRAINER.DACNET.WEIGHT_CON  # class compactness loss weight

        self.evaluator = build_evaluator(cfg, lab2cname=self.dm.lab2cname)
        self.num_samples = 0 # number of samples that cross the threshold
        self.total_num = 1e-8 # total number of samples in each epoch
        self.num_src_domains = len(self.cfg.DATASET.SOURCE_DOMAINS)
        # batch size for each source domain
        self.split_batch = self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE // self.num_src_domains
        
        # domain-level source attention weights for the fourth (last) residual block
        self.src_ca_last1 = [1. for _ in range(self.num_src_domains)]
        # domain-level target attention weights for the fourth (last) residual block
        self.tar_ca_last1 = [1.]
        # domain-level source attention weights for the third residual block
        self.src_ca_last2 = [1. for _ in range(self.num_src_domains)]
        self.tar_ca_last2 = [1.]

    def build_model(self):
        cfg = self.cfg
        print('Building model')
        assert 'ca' in cfg.MODEL.BACKBONE.NAME, 'Wrong backbone name {}. ' \
           'There must be ca (channel attention) in the backbone, e.g. resnet18_ca'.format(cfg.Model.BACKBONE.NAME)
        self.model = SimpleNet(cfg, cfg.MODEL, 0)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        fdim = self.model.fdim

        print('# params: {:,}'.format(count_num_param(self.model)))
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('model', self.model, self.optim, self.sched)

        self.classifier = torch.nn.Linear(fdim, self.num_classes)
        if cfg.MODEL.INIT_HEAD_WEIGHTS:
            load_pretrained_weights(self.classifier, cfg.MODEL.INIT_HEAD_WEIGHTS)
        else:
            try:
                load_pretrained_weights(self.classifier, cfg.MODEL.INIT_WEIGHTS)
            except:
                print('No wights found for classifier at {}'.format(cfg.MODEL.INIT_WEIGHTS))
        self.classifier.to(self.device)
        self.optim_c = build_optimizer(self.classifier, cfg.OPTIM)
        self.sched_c = build_lr_scheduler(self.optim_c, cfg.OPTIM)
        self.register_model('classifier', self.classifier, self.optim_c, self.sched_c)
    
    def forward_backward(self, batch_x, batch_u):
        global_step = self.batch_idx + self.epoch * self.num_batches
        ema_alpha = min(1 - 1 / (global_step+1), self.ema_alpha)
        parsed_data = self.parse_batch_train(batch_x, batch_u)
        input_x, input_x2, label_x, domain_x, input_u, input_u2 = parsed_data
        input_u = torch.cat([input_x, input_u], 0)
        input_u2 = torch.cat([input_x2, input_u2], 0)
        
        domain_x = torch.split(domain_x, self.split_batch, 0)
        domain_x = [d[0].item() for d in domain_x] # domain label for each batch of source samples
        
        # Generate artificial label
        with torch.no_grad():
            feat_weak, _ = self.model(input_u)
            output_u = F.softmax(self.classifier(feat_weak), 1)
            max_prob, label_u = output_u.max(1)
            mask_u = (max_prob >= self.conf_thre).float()

            self.num_samples += mask_u.sum()
            self.total_num += label_u.size()[0]

        # Supervised loss
        feat_x, src_ca = self.model(input_x)
        output_x = self.classifier(feat_x)
        loss_x = F.cross_entropy(output_x, label_x)

        # Unsupervised loss
        feat_u, tar_ca_strong = self.model(input_u2)
        output_u = self.classifier(feat_u)
        loss_u = F.cross_entropy(output_u, label_u, reduction='none')
        loss_u = (loss_u * mask_u).mean()

        loss = loss_x + loss_u * self.weight_u
        loss_summary = {
            'loss_x': loss_x.item(),
            'acc_x': compute_accuracy(output_x, label_x)[0].item(),
            'loss_u': loss_u.item(),
            'acc_u': compute_accuracy(output_u, label_u)[0].item()
        }

        # Domain attention consistency loss
        if self.weight_d > 0:
            loss_record = loss.item()
            # DAC loss for the fourth (last) residual block
            tar_ca1 = tar_ca_strong[-1] # get target attention weights
            _, tar_ca1 = torch.split(tar_ca1, [input_x2.size(0), input_u2.size(0) - input_x2.size(0)], 0)
            # do EMA to batch-level attention weights so that domain-level attention weights can be obtained
            mean_tar_ca1 = self.tar_ca_last1[0] * ema_alpha + (1. - ema_alpha) * torch.mean(tar_ca1, 0)
            self.tar_ca_last1[0] = mean_tar_ca1.detach()
            src_ca_tmp = torch.split(src_ca[-1], self.split_batch, 0)
            for ind, ca in enumerate(src_ca_tmp):
                domain_idx = domain_x[ind]
                mean_src_ca1 = self.src_ca_last1[domain_idx] * ema_alpha + (1. - ema_alpha) * torch.mean(ca, 0)
                self.src_ca_last1[ind] = mean_src_ca1.detach()
                loss += self.weight_d/self.num_src_domains * domain_discrepancy(mean_src_ca1, mean_tar_ca1, self.domain_loss_type)
            loss_summary['loss_dac_last'] = loss.item() - loss_record

            # DAC loss for the third residual block
            tar_ca2 = tar_ca_strong[-2] # get target attention weights
            _, tar_ca2 = torch.split(tar_ca2, [input_x2.size(0), input_u2.size(0) - input_x2.size(0)], 0)
            mean_tar_ca2 = self.tar_ca_last2[0] * ema_alpha + (1. - ema_alpha) * torch.mean(tar_ca2, 0)
            self.tar_ca_last2[0] = mean_tar_ca2.detach()
            src_ca_tmp = torch.split(src_ca[-2], self.split_batch, 0)
            for ind, ca in enumerate(src_ca_tmp):
                domain_idx = domain_x[ind]
                mean_src_ca2 = self.src_ca_last2[domain_idx] * ema_alpha + (1. - ema_alpha) * torch.mean(ca, 0)
                self.src_ca_last2[ind] = mean_src_ca2.detach()
                loss += self.weight_d/self.num_src_domains * domain_discrepancy(mean_src_ca2, mean_tar_ca2, self.domain_loss_type)
    
            loss_summary['loss_dac_all'] = loss.item() - loss_record

        # class compactness loss
        if self.weight_con > 0:
            with torch.no_grad():
                # we enforce class compactness loss only on these target samples that have consistent prediction under Guassian noise
                img_shape = input_u.shape
                img_noise = torch.randn(img_shape[0], img_shape[1], img_shape[2], img_shape[3]) * 0.15
                input_u3 = torch.flip(input_u, [3]) + img_noise.to(self.device)
                feat_u3, _ = self.model(input_u3)
                output_u3 = F.softmax(self.classifier(feat_u3), 1)
                max_prob3, label_u3 = output_u3.max(1)
                mask_u3 = (max_prob3 >= self.conf_thre).float()
                label_consist = (label_u == label_u3).float()
                mask_u = mask_u * mask_u3 * label_consist

            if mask_u.sum() > 0:
                weight = self.classifier.weight
                diff = mask_u.unsqueeze(1).expand_as(feat_u) * (feat_u - weight[label_u, :])
                loss_comp = self.weight_con * torch.mean(diff * diff)
            else:
                loss_comp = torch.tensor(0.)
            loss += loss_comp
            loss_summary['loss_cc'] = loss_comp.item()

        self.model_backward_and_update(loss)
      
        self.update_lr()

        return loss_summary

    @torch.no_grad()
    def test(self):
        """A generic testing pipeline."""
        # display samples that have predicted probability > threshold
        print('samples above the threshold {}({}/{})'.format(
            float(self.num_samples ) / self.total_num, self.num_samples, self.total_num))
        self.num_samples = 0
        self.total_num = 0

        self.set_model_mode('eval')
        self.evaluator.reset()
        
        split = self.cfg.TEST.SPLIT
        print('Do evaluation on {} set'.format(split))
        data_loader = self.val_loader if split == 'val' else self.test_loader
        assert data_loader is not None
        
        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output, _ = self.model_inference(input)
            output = self.classifier(output)
            self.evaluator.process(output, label)
            
        results = self.evaluator.evaluate()
            
        for k, v in results.items():
            tag = '{}/{}'.format(split, k)
            self.write_scalar(tag, v, self.epoch)

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x['img']
        input_x2 = batch_x['img2']
        label_x = batch_x['label']
        domain_x = batch_x['domain']
        input_u = batch_u['img']
        input_u2 = batch_u['img2']

        input_x = input_x.to(self.device)
        input_x2 = input_x2.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)
        input_u2 = input_u2.to(self.device)

        return input_x, input_x2, label_x, domain_x, input_u, input_u2


