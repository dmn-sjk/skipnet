from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import os
import time
import logging
from collections import OrderedDict
import yaml
from torch.distributions import Categorical

import models
from datasets.data_loading import get_dataloader
from utils.config import get_config, save_config
from utils.utils import save_checkpoint, get_save_path, set_seed
from utils.metrics import AverageMeter, accuracy, save_final_metrics, ListAverageMeter
from datasets.cifar_c_dataset import CORRUPTIONS
from utils.modules import BatchCrossEntropy


def main():
    raise NotImplementedError()
    args = get_config()
    save_path = args.save_path = get_save_path(args)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    set_seed(args.seed)

    # config logging file
    args.logger_file = os.path.join(save_path, 'log_{}.txt'.format(args.cmd))
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)
    
    save_config(args, os.path.join(save_path, 'config.yaml'))
    
    # create model
    model = models.__dict__[args.arch](args.pretrained)
    model = torch.nn.DataParallel(model).cuda()
    
    criterion = nn.CrossEntropyLoss().cuda()
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['iter']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (iter: {})'.format(
                args.resume, checkpoint['iter']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    if args.dataset[-1] != 'c':
        raise NotImplementedError('Only corrupted for now')
    
    res_dict = OrderedDict()
    yaml.add_representer(OrderedDict, lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))

    args.severity = 5
    for i, corruption in enumerate(CORRUPTIONS):
        args.corruption = corruption
        train_loader = get_dataloader(args, split='train')
        test_loader = get_dataloader(args, split='val')
        
        logging.info('start training {} - {} - {}'.format(args.arch, args.corruption, args.severity))
        
        curr_task_acc, _ = run_training(args, model, i)
        
        task_key = f'task_{i} - ({args.corruption}_{args.severity})'
        
        res_dict[task_key] = {
            'curr_acc': curr_task_acc
        }

        # evaluate on previous domains
        for j, val_corr in enumerate(CORRUPTIONS[:i]):
            args.corruption = val_corr
            test_loader = get_dataloader(args, split='val')
            acc, _ = validate(args, test_loader, model, criterion)
            res_dict[task_key][f'task_{j}_acc'] = acc
        
        save_final_metrics(res_dict, save_path=os.path.join(args.save_path, 'metric.yaml'))