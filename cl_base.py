""" This file is for training original model without routing modules.
"""

from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import os
import time
import logging
from collections import OrderedDict
import yaml

import models
from datasets.data_loading import get_dataloader
from utils.config import get_config, save_config
from utils.utils import save_checkpoint, get_save_path, set_seed
from utils.metrics import AverageMeter, accuracy, save_final_metrics
from datasets.cifar_c_dataset import CORRUPTIONS


def main():
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

    args.severity = 5
    for i, corruption in enumerate(CORRUPTIONS):
        args.corruption = corruption
        logging.info('start training {} - {} - {}'.format(args.arch, args.corruption, args.severity))
        curr_task_acc, best_model_path = run_training(args, model, i)
        
        res_dict[i] = {
            'domain': f'{args.corruption}_{args.severity}', 
            'curr_acc': curr_task_acc
        }

        # load best model 
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['state_dict'])

        # evaluate on previous domains
        for j, val_corr in enumerate(CORRUPTIONS[:i]):
            args.corruption = val_corr
            test_loader = get_dataloader(args, split='val')
            acc, _ = validate(args, test_loader, model, criterion)
            res_dict[i][f'task_{j}_acc'] = acc
            res_dict[i][f'task_{j}_acc_diff_init'] = acc - res_dict[j]['curr_acc']
        
        save_final_metrics(res_dict, save_path=os.path.join(args.save_path, 'metric.yaml'))

def run_training(args, model, task_id):
    best_prec1 = 0
    best_loss = float('inf')

    train_loader = get_dataloader(args, split='train')
    test_loader = get_dataloader(args, split='val')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for i in range(args.start_iter, args.iters):
        model.train()

        input, target = next(iter(train_loader))
        # measuring data loading time
        data_time.update(time.time() - end)

        target = target.squeeze().long().cuda(non_blocking=True)
        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print log
        if i % args.print_freq == 0:
            logging.info("Task: {0}\t"
                         "Iter: [{1}/{2}]\t"
                         "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                         "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                         "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                         "Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t".format(
                            task_id,
                            i,
                            args.iters,
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses,
                            top1=top1)
            )

        if (i % args.eval_every == 0 and i > 0) or (i == args.iters - 1):
            prec1, loss = validate(args, test_loader, model, criterion)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            checkpoint_path = os.path.join(args.save_path,
                                           'checkpoint_{:05d}.pth.tar'.format(
                                               i))
            if is_best:
                best_model_path = save_checkpoint({
                    'iter': i,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                },
                    is_best, filename=checkpoint_path)
            
            # 2. Early stopping
            if loss < (best_loss - 0.001):
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            patience = 5
            if patience_counter >= patience:
                logging.info("Early stopping stop!")
                break


    return best_prec1, best_model_path

def validate(args, test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluation mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target.squeeze().long().cuda(non_blocking=True)
        input_var = input
        target_var = target

        # compute output
        with torch.no_grad():
            output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        top1.update(prec1.item(), input.size(0))
        losses.update(loss.data.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    # logging.info(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg, losses.avg


def test_model(args):
    # create model
    model = models.__dict__[args.arch](args.pretrained)
    model = torch.nn.DataParallel(model).cuda()

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

    test_loader = get_dataloader(args, split='test')

    criterion = nn.CrossEntropyLoss().cuda()

    validate(args, test_loader, model, criterion)

if __name__ == '__main__':
    main()
