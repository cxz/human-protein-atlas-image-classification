import os
import argparse
import json
import uuid
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


import torch.backends.cudnn as cudnn
import torch.backends.cudnn

# from loss import LossBinary, FocalLoss, LossLovasz, BCEDiceJaccardLoss, LossHinge
import losses

import dataset
import models
import utils
import losses
from validation import validation_multi


def build_train_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--name', type=str)
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--fold', type=int, help='fold', default=0)
    arg('--output-dir', default='../data/runs', help='checkpoint root')
    arg('--batch-size', type=int, default=32)
    arg('--iter-size', type=int, default=1)
    arg('--n-epochs', type=int, default=150)
    arg('--lr', type=float, default=0.001)
    arg('--workers', type=int, default=8)
    arg('--seed', type=int, default=0)
    arg('--model', type=str, default='resnet18', choices=models.archs)
    arg('--loss', type=str, default='multi', choices=['multi'])
    arg('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    arg('--focal-gamma', type=float, default=1)  # .5
    arg('--num-channels', type=int, default=4)
    arg('--weighted-sampler', action="store_true", dest='weighted_sampler')
    arg('--no-weighted-sampler', action="store_false", dest="weighted_sampler")
    arg('--resume', action="store_true")
    arg('--load-weights', action="store_true")
    parser.set_defaults(weighted_sampler=False)
    args = parser.parse_args()
    return args


def main():
    args = build_train_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    experiment = args.name if args.name else uuid.uuid4().hex

    output_dir = Path(args.output_dir) / experiment
    output_dir.mkdir(exist_ok=True, parents=True)
    output_dir.joinpath('params.json').write_text(json.dumps(vars(args), indent=True, sort_keys=True))

    # this is different than --resume, which loads optimizer state as well
    initial_model_path = os.path.join(output_dir, f"model_{args.fold}.pth") if args.load_weights else None
    model = models.get_model(initial_model_path, args.model)

    train_ids, val_ids = dataset.get_split(args.fold)

    cudnn.benchmark = True

    train_loader = dataset.make_loader(
        train_ids,
        num_channels=args.num_channels,
        transform=dataset.train_transform(),
        shuffle=True,
        weighted_sampling=args.weighted_sampler,
        batch_size=args.batch_size,
        workers=args.workers)

    valid_loader = dataset.make_loader(
        val_ids,
        num_channels=args.num_channels,
        transform=dataset.val_transform(),
        shuffle=False,
        weighted_sampling=False,
        batch_size=args.batch_size,  # len(device_ids),
        workers=args.workers)

#     train last layer only
#     for m in model.children():
#         if m in [model.fc]:
#             continue
#         for param in m.parameters():
#             param.requires_grad = False

    # set all layers except fc to lr=0
    #fc_params = list(map(id, model.fc.parameters()))
    #base_params = filter(lambda p: id(p) not in fc_params, model.parameters())
    #optimizer = torch.optim.Adam([
    #     {'params': base_params, 'lr': args.lr * 0.001},
    #     {'params': model.fc.parameters(), 'lr': args.lr}
    #], lr=args.lr * 0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)

    # criterion = FocalLoss(gamma=args.focal_gamma)
    criterion = nn.BCEWithLogitsLoss().cuda()
    # criterion = losses.f1_loss
        
    validation_fn = validation_multi
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #    optimizer, mode='max', verbose=True, min_lr=1e-7, factor=0.5, patience=5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    snapshot = utils.fold_snapshot(output_dir, args.fold) if args.resume else None

    device_ids = list(map(int, args.device_ids.split(','))) if args.device_ids else None
    wrapped_model = nn.DataParallel(model, device_ids=device_ids).cuda()

    # unfreeze
    # for m in model.children():
    #     for param in m.parameters():
    #         param.requires_grad = True

    # fc1_params = list(map(id, model.fc1.parameters()))
    # fc2_params = list(map(id, model.fc2.parameters()))
    # base_params = filter(lambda p: id(p) not in fc1_params and id(p) not in fc2_params, model.parameters())
    #
    # # fc at args.lr and rest with args.lr * 0.1
    # optimizer = torch.optim.Adam([
    #     {'params': base_params},
    #     {'params': model.fc1.parameters(), 'lr': args.lr},
    #     {'params': model.fc2.parameters(), 'lr': args.lr},
    # ], lr=args.lr * 0.1)

    # aternatively, add the unfrozen fc2 weight to the current optimizer
    # optimizer.add_param_group({'params': net.fc2.parameters()})

    utils.train(
        experiment=experiment,
        output_dir=output_dir,
        optimizer=optimizer,
        args=args,
        model=wrapped_model,
        criterion=criterion,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation_fn=validation_fn,
        fold=args.fold,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        snapshot=snapshot,
        iter_size=args.iter_size
    )

if __name__ == '__main__':
    main()
