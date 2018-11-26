""" merge predictions and generate submission.
"""

import os
import argparse
import json
import pickle
import datetime

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import torch
import csv

import predict
import dataset
import models
import validation
import utils

from torch.utils.data import DataLoader
from torch.nn import functional as F
from albumentations import Compose, Resize, HorizontalFlip, VerticalFlip, OneOf, RandomBrightness, RandomContrast, RandomGamma, GaussNoise, Blur, ElasticTransform, ShiftScaleRotate, Normalize


def generate_submission(out_csv, preds):    
    sample_df = pd.read_csv('../input/sample_submission.csv')
    test_ids = sample_df.Id.values
    
    def format(p):        
        return ' '.join([str(x[0]) for x in np.argwhere(p)])
    
    rows = [(id_, format(p)) for id_, p in zip(test_ids, preds)]
    sub = pd.DataFrame(rows, columns=['Id', 'Predicted'])
    sub.to_csv(out_csv, index=False)


def evaluate(oof_pred):        
    train_df = pd.read_csv(os.path.join('../input', 'train.csv'))
    target = []
    
    for id_, label in train_df[['Id', 'Target']].values:
        y = [0] * dataset.NUM_CLASSES
        for i in label.split(' '): y[int(i)] = 1
        target.append(y)
        
    target = np.array(target, np.uint8)            
    valid_thresholds, valid_f1 = validation.f1_macro(target, oof_pred, debug=False)
    return valid_thresholds, valid_f1   

def main(write_submission=True):
    experiments = {
        '../data/runs/resnet50-s224-v2': 1,        
    }
    
    train_df = pd.read_csv(os.path.join(dataset.PATH, 'train.csv'))
    preds = np.zeros((11702, 28), dtype=np.float32)
    
    for exp, weight in experiments.items():
        print(f'processing experiment {exp}')
        
        #
        # load validation & test set predictions
        #
        folds = list(range(5))
        oof_pred = np.zeros((31072, 28), dtype=np.float32)
        fold_preds = []        
        for fold in range(5):
            _, val_ids = dataset.get_split(fold)    
            val_idx = np.array(train_df[train_df.Id.isin(val_ids)].index)
            oof_pred[val_idx] = np.load(os.path.join(exp, f"val_preds_fold{fold}.npy"))
            fold_preds.append(np.load(os.path.join(exp, f"test_preds_fold{fold}.npy")))
                                
        # find threshold per class using oof 
        val_thresholds, val_f1 = evaluate(oof_pred)
        print('blended val_f1: ', val_f1)
                
        # folds mean        
        fold_preds = np.sum(fold_preds, axis=0) / len(folds)
        
        # add thresholded with weights to final array
        for klazz, klazz_thr in val_thresholds.items():
            preds[:, klazz] += weight * (fold_preds[:, klazz] > klazz_thr)

    # todo: replace with majority voting
    final = np.round(1. * preds / sum(experiments.values()))
    np.save('preds.npy', final)

    if write_submission:
        output_csv = f'../submissions/{datetime.datetime.now().strftime("%Y%m%d%H%M")}.csv'
        print('writing to ', output_csv)
        
        generate_submission(output_csv, final)
        print('done.')
    
if __name__ == '__main__':
    main()
