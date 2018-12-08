import os
import json
from datetime import datetime
from pathlib import Path

import random
import numpy as np
import pandas as pd

import torch
import tqdm
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import SGD, Adam
import sklearn

import validation


def fold_snapshot(output_dir, fold):
    fname = os.path.join(output_dir, f"model_{fold}.pth")
    return fname if os.path.exists(fname) else None


def cuda(x):
    return x.cuda(async=True) if torch.cuda.is_available() else x


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save(model, optimizer, model_path, epoch, step, valid_best):
    # epoch_path = "{}_epoch{}.pth".format(str(model_path), epoch)
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'valid_best': valid_best,
        'optimizer': optimizer.state_dict(),
        'step': step,
    }, str(model_path))


# evaluate meters
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(experiment, output_dir, args,
          model, criterion, scheduler, train_loader, valid_loader, validation_fn, optimizer,
          n_epochs=None, fold=None, batch_size=None, snapshot=None, iter_size=1, 
          extra_valid_loaders={},
          val_metric='val_f1'):

    # TODO: refactoring -- move out
    if snapshot:
        state = torch.load(snapshot)
        epoch = state['epoch']
        step = state['step']
        valid_best = state['valid_best']
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])  # causing oom if history too large
        # set_learning_rate(optimizer, 1e-5)
        print('Restored model, fold{}, epoch {}, step {:,}, valid_best {}'.format(fold, epoch, step, valid_best))

        #
        # optimizer = Adam(model.parameters(), lr=1e-5)
        # scheduler = ReduceLROnPlateau(optimizer, verbose=True, patience=10, min_lr=1e-7, factor=0.5)
        del state

    else:
        epoch = 1
        step = 0
        valid_best = None

    model_path = output_dir / 'model_{fold}.pth'.format(fold=fold)

    scores_fname = output_dir / 'scores.csv'
    scores = pd.read_csv(scores_fname).values.tolist() if scores_fname.exists() else []

    steps_per_epoch = len(train_loader) * batch_size
    smooth_mean = 10

    for epoch in range(epoch, n_epochs + 1):
        model.train()
        tq = tqdm.tqdm(total=(steps_per_epoch))
        lr = get_learning_rate(optimizer)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []

        epoch_f1 = AverageMeter()
        epoch_losses = AverageMeter()
        debug_epoch_targets = None
        debug_epoch_preds = None

        tl = train_loader
        try:
            mean_loss = 0
            optimizer.zero_grad()
            batch_loss_value = 0

            for i, (inputs, targets) in enumerate(tl):
                inputs = cuda(inputs)
                
                if targets.size(0) == 1:
                    continue # invalid batch

                with torch.no_grad():
                    targets = cuda(targets)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                targets_npy = targets.cpu()
                outputs_npy = outputs.detach().sigmoid().cpu()

                epoch_losses.update(loss.item(), targets.size(0))
                epoch_f1.update(sklearn.metrics.f1_score(targets, outputs_npy > 0.15, average='macro'))

                if i == 0:
                    debug_epoch_targets = targets_npy
                    debug_epoch_preds = outputs_npy
                else:
                    debug_epoch_targets = np.concatenate([debug_epoch_targets, targets_npy])
                    debug_epoch_preds = np.concatenate([debug_epoch_preds, outputs_npy])


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # # acumulate gradient over iter_size batches
                # loss.backward()
                # batch_loss_value += loss.detach().cpu().numpy()
                #
                # # accumulate gradient for n iters
                # if i % iter_size == 0:
                #     optimizer.step()
                #     batch_loss_value /= iter_size
                #     losses.append(batch_loss_value.item())
                #     mean_loss = np.mean(losses[-smooth_mean:])
                #
                #     batch_loss_value = 0
                #     optimizer.zero_grad()

                step += 1

                tq.update(targets.size(0))
                tq.set_postfix(loss='{:.4f}'.format(epoch_losses.avg),
                               f1='{:.4f}'.format(epoch_f1.avg))

            #
            # epoch end.
            #
            _, train_f1_mean = validation.f1_macro(debug_epoch_targets, debug_epoch_preds, debug=False)
            print('train f1: ', train_f1_mean)
            tq.close()

            valid_metrics = validation_fn(model, criterion, valid_loader)
            scores.append([
                "{:01d}".format(fold),
                "{:03d}".format(epoch),
                "{:.4f}".format(valid_metrics['val_loss']),
                "{:.4f}".format(valid_metrics[val_metric])
            ])
            scores_df = pd.DataFrame(scores, columns=['fold', 'epoch', 'val_loss', val_metric])
            scores_df.to_csv(str(scores_fname), index=False)
            
            for name, extra_loader in extra_valid_loaders.items():
                valid_metrics = validation_fn(model, criterion, extra_loader)
                print(name, valid_metrics)

            if valid_best is None or valid_metrics[val_metric] > valid_best:
                valid_best = valid_metrics[val_metric]
                save(model, optimizer, model_path, epoch, step, valid_best)

            # use metric for lr scheduler
            scheduler.step(valid_metrics[val_metric]) 

        except KeyboardInterrupt:
            tq.close()
            # print('Ctrl+C, saving snapshot')
            # save(model, optimizer, model_path, epoch, step, valid_best)
            print('done.')
            return
