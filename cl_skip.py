from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import os
import time
import logging
from collections import OrderedDict
from torch.distributions import Categorical

import models
from datasets.data_loading import get_dataloader
from utils.config import get_config, save_config
from utils.utils import save_checkpoint, get_save_path, set_seed
from utils.metrics import AverageMeter, accuracy, save_final_metrics, ListAverageMeter
from datasets.cifar_c_dataset import CORRUPTIONS
from utils.modules import BatchCrossEntropy


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
        curr_task_acc, _ = run_training(args, model, i)
        
        res_dict[i] = {
            'domain': f'{args.corruption}_{args.severity}', 
            'curr_acc': curr_task_acc
        }

        # evaluate on previous domains
        # for j, val_corr in enumerate(CORRUPTIONS[:i]):
        for j, val_corr in enumerate(CORRUPTIONS):
            args.corruption = val_corr
            test_loader = get_dataloader(args, split='val')
            acc, _ = validate(args, test_loader, model, criterion)
            res_dict[i][f'task_{j}_acc'] = acc
            if j in res_dict.keys():
                res_dict[i][f'task_{j}_acc_diff_init'] = acc - res_dict[j]['curr_acc']
        
        save_final_metrics(res_dict, save_path=os.path.join(args.save_path, 'metric.yaml'))

def run_training(args, model, task_id):
    train_loader = get_dataloader(args, split='train')
    test_loader = get_dataloader(args, split='val')
    
    # if task_id == 0:
    acc, best_model_path = train_sp(args, model, task_id, train_loader, test_loader)
    # load best model 
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['state_dict'])

    acc, best_model_path = train_rl(args, model, task_id, train_loader, test_loader)
    # load best model 
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    # else:
    #     acc, best_model_path = gated_train(args, model, task_id, train_loader, test_loader)
    #     # load best model 
    #     checkpoint = torch.load(best_model_path)
    #     model.load_state_dict(checkpoint['state_dict'])

    # best_model_path = 'save_checkpoints/cl/cifar10_rnn_gate_rl_38/cifar10c/test/model_best.pth.tar'
    # checkpoint = torch.load(best_model_path)
    # model.load_state_dict(checkpoint['state_dict'])

    # model.eval()
    # all_masks = [torch.empty(0) for i in range(17)]
    # for i, (input, target) in enumerate(train_loader): 
    #     input_var = Variable(input)
    #     with torch.no_grad():
    #         output, masks, _ = model(input_var)

    #     for j, mask in enumerate(masks):
    #         all_masks[j] = torch.cat((all_masks[j], mask.cpu()), dim=0)
    
    # for j, mask in enumerate(all_masks):
    #     # all_masks[j] = torch.sum(mask, dim=0) > mask.shape[0]
    #     all_masks[j] = torch.mean(mask, dim=0)
        
    #     checkpoint_path = os.path.join(args.save_path,
    #                                     'checkpoint_{:05d}.pth.tar'.format(
    #                                         i))
    # m = torch.stack(all_masks, dim=0)
    # save_path = os.path.join(os.path.dirname(checkpoint_path), str(task_id) + '_' + 'avg_mask.pth')    
    # torch.save(m, save_path)
    
    # acc = 0

    return acc, best_model_path

def gated_train(args, model, task_id, train_loader, test_loader):
    best_prec1 = 0
    best_loss = float('inf')
    
    model.requires_grad_(True)
    from models import RNNGatePolicy
    for module in model.modules():
        if isinstance(module, RNNGatePolicy):
            module.requires_grad_(False)
            print('Freezing gating policy!')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], 
                                args.lr_sp,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    skip_ratios = ListAverageMeter()

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
        output, masks, _ = model(input_var)

        # collect skip ratio of each layer
        skips = [mask.data.le(0.5).float().mean() for mask in masks]
        if skip_ratios.len != len(skips):
            skip_ratios.set_len(len(skips))
        
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        skip_ratios.update(skips, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # repackage hidden units for RNN Gate
        # if args.gate_type == 'rnn':
        #     model.module.control.repackage_hidden()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print log
        if i % args.print_freq == 0 or i == (args.iters - 1):
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
                    is_best, filename=checkpoint_path
                    # , name_prefix=str(task_id)
                    )
            
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

def train_sp(args, model, task_id, train_loader, test_loader):
    best_prec1 = 0
    best_loss = float('inf')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr_sp,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    skip_ratios = ListAverageMeter()

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
        output, masks, _ = model(input_var)

        # collect skip ratio of each layer
        skips = [mask.data.le(0.5).float().mean() for mask in masks]
        if skip_ratios.len != len(skips):
            skip_ratios.set_len(len(skips))
        
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        skip_ratios.update(skips, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # repackage hidden units for RNN Gate
        # if args.gate_type == 'rnn':
        #     model.module.control.repackage_hidden()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print log
        if i % args.print_freq == 0 or i == (args.iters - 1):
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
                    is_best, filename=checkpoint_path
                    # , name_prefix=str(task_id)
                    )
            
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

def train_rl(args, model, task_id, train_loader, test_loader):
    best_prec1 = 0
    best_loss = float('inf')

    # define loss function (criterion) and optimizer
    criterion = BatchCrossEntropy().cuda()
    total_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr_rl,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    # extract gate actions and rewards
    if args.gate_type == 'ff':
        gate_saved_actions = model.module.saved_actions
        gate_rewards = model.module.rewards
    elif args.gate_type == 'rnn':
        gate_saved_actions = model.module.control.saved_actions
        gate_rewards = model.module.control.rewards
        
    # clear saved actions and rewards
    del gate_saved_actions[:]
    del gate_rewards[:]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_rewards = AverageMeter()
    total_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    skip_ratios = ListAverageMeter()

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
        output, masks, probs = model(input_var)

        skips = [mask.data.le(0.5).float().mean() for mask in masks]
        if skip_ratios.len != len(skips):
            skip_ratios.set_len(len(skips))

        pred_loss = criterion(output, target_var)

        # re-weight gate rewards
        normalized_alpha = args.alpha / len(gate_saved_actions)
        # intermediate rewards for each gate
        for act in gate_saved_actions:
            gate_rewards.append((1 - act.float()).data * normalized_alpha)
        # pdb.set_trace()
        # collect cumulative future rewards
        R = - pred_loss.data
        cum_rewards = []
        for r in gate_rewards[::-1]:
            R = r + args.gamma * R
            cum_rewards.insert(0, R)

        # apply REINFORCE to each gate
        # Pytorch 2.0 version. `reinforce` function got removed in Pytorch 3.0
        rl_losses = 0
        for action, prob, R in zip(gate_saved_actions, probs, cum_rewards):
            # action.reinforce(args.rl_weight * R)
            dist = Categorical(prob)
            _loss = -dist.log_prob(action)*R
            rl_losses += _loss
        rl_losses = rl_losses.mean()

        total_loss = total_criterion(output, target_var) + rl_losses * args.rl_weight

        optimizer.zero_grad()
        # optimize hybrid loss
        # torch.autograd.backward(gate_saved_actions + [total_loss])
        total_loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        total_rewards.update(cum_rewards[0].mean(), input.size(0))
        total_losses.update(total_loss.mean().data.item(), input.size(0))
        losses.update(pred_loss.mean().data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        skip_ratios.update(skips, input.size(0))
        total_gate_reward = sum([r.mean() for r in gate_rewards])

        # clear saved actions and rewards
        del gate_saved_actions[:]
        del gate_rewards[:]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print log
        if i % args.print_freq == 0 or i == (args.iters - 1):
            logging.info("Task: {0}\t"
                         "Iter: [{1}/{2}]\t"
                         "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                         "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                         "Total reward {total_rewards.val: .3f}"
                         "({total_rewards.avg: .3f})\t"
                         "Total gate reward {total_gate_reward: .3f}\t"
                         "Total Loss {total_losses.val:.3f} "
                         "({total_losses.avg:.3f})\t"
                         "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                         "Acc@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                            task_id,
                            i,
                            args.iters,
                            batch_time=batch_time,
                            data_time=data_time,
                            total_rewards=total_rewards,
                            total_gate_reward=total_gate_reward,
                            total_losses=total_losses,
                            loss=losses,
                            top1=top1)
            )

        if (i % args.eval_every == 0 and i > 0) or (i == args.iters - 1):
            prec1, loss = validate(args, test_loader, model, total_criterion)
            
            # clear saved actions and rewards
            del gate_saved_actions[:]
            del gate_rewards[:]
            
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
                    is_best, filename=checkpoint_path
                    # , name_prefix=str(task_id)
                    )
            
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
    skip_ratios = ListAverageMeter()

    # switch to evaluation mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target.squeeze().long().cuda(non_blocking=True)
        input_var = input
        target_var = target

        # compute output
        with torch.no_grad():
            output, masks, _ = model(input_var)
        skips = [mask.data.le(0.5).float().mean() for mask in masks]
        if skip_ratios.len != len(skips):
            skip_ratios.set_len(len(skips))
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        top1.update(prec1.item(), input.size(0))
        skip_ratios.update(skips, input.size(0))
        losses.update(loss.data.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
    
    skip_summaries = []
    for idx in range(skip_ratios.len):
        # logging.info(
        #     "{} layer skipping = {:.3f}".format(
        #         idx,
        #         skip_ratios.avg[idx],
        #     )
        # )
        skip_summaries.append(1-skip_ratios.avg[idx])
    # compute `computational percentage`
    cp = ((sum(skip_summaries) + 1) / (len(skip_summaries) + 1)) * 100
    logging.info('*** Computation Percentage: {:.3f} %'.format(cp))

    return top1.avg, losses.avg



if __name__ == '__main__':
    main()
