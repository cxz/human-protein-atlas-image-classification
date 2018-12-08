"""
"""

import os
import argparse
import json
import pickle

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import torch

import predict
import dataset
import models
import validation
import utils

from torch.utils.data import DataLoader
from torch.nn import functional as F
from albumentations import Compose, Resize, HorizontalFlip, VerticalFlip, OneOf, RandomBrightness, RandomContrast, RandomGamma, GaussNoise, Blur, ElasticTransform, ShiftScaleRotate, Normalize


def build_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path', type=str, default='experiment folder', help='')
    arg('--batch-size', type=int, default=32)
    arg('--fold', type=int, default=-1, choices=[0, 1, 2, 3, 4, -1], help='-1: all folds')    
    arg('--workers', type=int, default=8)
    arg('--tta', type=int, default=5)
    args = parser.parse_args()
    return args


def predict(model, ids, transform, kind, batch_size=32):
    loader = dataset.make_loader(
        ids,
        shuffle=False,
        mode=kind,
        batch_size=batch_size,
        transform=transform)
        
    preds = np.zeros((len(ids), dataset.NUM_CLASSES), dtype=np.float32)
    pred_idx = 0
    with torch.no_grad():
        for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='predict')):
            inputs = utils.cuda(inputs)
            outputs = model(inputs)            
            outputs_npy = torch.sigmoid(outputs).cpu().numpy()
            for p in outputs_npy:
                preds[pred_idx] = p
                pred_idx += 1
    return preds


def predict_tta(model, ids, output, kind='test', batch_size=32, n_tta=5):
    size = dataset.SIZE
    base_transform = dataset.val_transform()
    
    preds = np.zeros((1+n_tta, len(ids), dataset.NUM_CLASSES), dtype=np.float32)
    preds[0] = predict(model, ids, transform=base_transform, kind=kind, batch_size=batch_size)
            
    tta_transform = dataset.train_transform()
    
    for tta_idx in range(1, n_tta):
        preds[tta_idx] = predict(model, ids, transform=tta_transform, kind=kind, batch_size=batch_size)
        
    mean_preds = np.mean(preds, axis=0)
    np.save(output, mean_preds)


def eval_fold(experiment_path, fold):
    model_path = os.path.join(experiment_path, f"model_{fold}.pth")    
    _, val_ids = dataset.get_split(fold)        
    with open(os.path.join(experiment_path, f"val_preds_fold{fold}.npy"), "rb") as f:
        val_preds = np.load(f)
        
    train_df = pd.read_csv(os.path.join('../input', 'train.csv'))
    val_target = {}
    
    for id_, label in train_df[['Id', 'Target']].values:
        y = [0] * dataset.NUM_CLASSES
        for i in label.split(' '): y[int(i)] = 1
        val_target[id_] = y
        
    targets = np.array([val_target[id_] for id_ in val_ids], np.uint8)            
    valid_thresholds, valid_f1 = validation.f1_macro(targets, val_preds, debug=False)
    
    #with open(f"val_preds_fold{fold}_thresholds.json", 'w') as f:
    #    f.write(json.dumps(valid_thresholds, indent=4))        

    return valid_thresholds, valid_f1    
    
if __name__ == '__main__':

    args = build_args()

    with open(os.path.join(args.path, 'params.json'), 'r') as f:
        config = json.loads(f.read())
        
    model_type = config['model']

    test_ids = dataset.get_test_ids()
    
    folds = list(range(5)) if args.fold == -1 else [args.fold]
    for fold in folds:
        print('processing fold ', fold)
        model_path = os.path.join(args.path, f"model_{fold}.pth")
        model = models.get_model(model_path, model_type=model_type)
        model.eval()
        print(f'{model_path} loaded.')

        print('predicting val set')
        val_output = os.path.join(args.path, f"val_preds_fold{fold}.npy")
        _, val_ids = dataset.get_split(fold)
        predict_tta(model, val_ids, val_output, kind='val', n_tta=args.tta)

        valid_thresholds, valid_f1 = eval_fold(args.path, fold)
        print(f'fold {fold}: {valid_f1}')

        print('predicting test set')
        test_output = os.path.join(args.path, f"test_preds_fold{fold}.npy")        
        predict_tta(model, test_ids, test_output, kind='test', batch_size=args.batch_size, n_tta=args.tta)
