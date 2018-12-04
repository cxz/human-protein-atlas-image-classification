import numpy as np
import sklearn

import torch

import utils
from sklearn.metrics import f1_score, precision_score, recall_score


from collections import Counter
import pickle


def f1_macro(y_true, y_pred, debug=True):
    """ Calculate f1-score for ohe"""

    klazz_f1 = {}
    klazz_f1_thr = {}
    for klazz in range(y_true.shape[1]):
        p = []
        r = []
        f1 = []
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for th in thresholds:
            y_true_ = y_true[:, klazz]
            y_pred_ = (y_pred[:, klazz] > th).astype(np.uint8)
            f1_ = f1_score(y_true_, y_pred_)
            f1.append(f1_)
        klazz_f1[klazz] = np.max(f1)
        klazz_f1_thr[klazz] = thresholds[np.argmax(f1)]
        if debug:
            print(f"class={klazz}, f1={np.max(f1)}, {f1}")
    return klazz_f1, np.mean(list(klazz_f1.values()))


def validation_multi(model, criterion, valid_loader):
    # stabilize batchnormal layers
    # for batch_idx, (inputs, targets) in enumerate(valid_loader):
    #     model(inputs)
    #     if batch_idx >= 10:
    #         break
        
    with torch.no_grad():
        model.eval()
        losses = []
        f1_scores = []

        # collect targets and predictions for whole validations set
        targets_npy = []
        outputs_npy = []

        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            inputs = utils.cuda(inputs)
            with torch.no_grad():
                targets = utils.cuda(targets)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())

            targets_npy.extend(targets.cpu().numpy())
            outputs_npy.extend(torch.sigmoid(outputs).cpu().numpy())

            # print(losses)

        with open('../data/tmp/val.pkl', 'wb') as f:
            pickle.dump([targets_npy, outputs_npy], f)

        targets_npy = np.squeeze(np.array(targets_npy).astype(np.uint8))
        outputs_npy = np.squeeze(np.array(outputs_npy).astype(np.float32))

        valid_loss = np.mean(losses)
        _, valid_f1 = f1_macro(targets_npy, outputs_npy, debug=False)
        valid_acc = (targets_npy == (outputs_npy > 0.15).astype(np.uint8)).mean()

        print("valid loss: {:.4f}, valid_f1: {:.4f}, valid_acc: {:.4f}".format(valid_loss, valid_f1, valid_acc))
        metrics = {'val_loss': valid_loss, 'val_f1': valid_f1, 'val_acc': valid_acc}
        return metrics


def test_f1_macro():
    y_true = np.array([[0, 0, 1], [0, 1, 1]])
    y_pred = np.array([[0, 0, 1], [0, 1, 0]])
    f1_macro(y_true, y_pred)
    print('==')
    print(y_true[:, 2])


def acc(preds, targets, th=0.0):
    preds = (preds > th).astype(np.uint8)
    targets = targets.astype(np.uint8)
    return (preds == targets).mean()

if __name__ == '__main__':
    y_true = np.array([[0, 0, 1], [0, 1, 1]])
    y_pred = np.array([[0, 0, 1], [0, 1, 0]])
    print(acc(y_true, y_pred))

