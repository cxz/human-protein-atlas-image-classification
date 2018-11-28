import os
import itertools
from collections import Counter

import torch
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import KFold

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from albumentations import Compose, Resize, HorizontalFlip, VerticalFlip, OneOf, RandomBrightness, RandomContrast, RandomGamma, GaussNoise, Blur, ElasticTransform, ShiftScaleRotate, Normalize
from albumentations.torch.functional import img_to_tensor

SIZE = 512
PATH = '../input'
TRAIN_PATH = os.path.join(PATH, f"train-{SIZE}") if SIZE != 512 else os.path.join(PATH, f"train")
TEST_PATH = os.path.join(PATH, f"test-{SIZE}") if SIZE != 512 else os.path.join(PATH, f"test")
DATA = '../data'
NUM_CLASSES = 28

label_names = {
    0:  'Nucleoplasm',
    1:  'Nuclear membrane',
    2:  'Nucleoli',
    3:  'Nucleoli fibrillar center',
    4:  'Nuclear speckles',
    5:  'Nuclear bodies',
    6:  'Endoplasmic reticulum',
    7:  'Golgi apparatus',
    8:  'Peroxisomes',
    9:  'Endosomes',
    10:  'Lysosomes',
    11:  'Intermediate filaments',
    12:  'Actin filaments',
    13:  'Focal adhesion sites',
    14:  'Microtubules',
    15:  'Microtubule ends',
    16:  'Cytokinetic bridge',
    17:  'Mitotic spindle',
    18:  'Microtubule organizing center',
    19:  'Centrosome',
    20:  'Lipid droplets',
    21:  'Plasma membrane',
    22:  'Cell junctions',
    23:  'Mitochondria',
    24:  'Aggresome',
    25:  'Cytosol',
    26:  'Cytoplasmic bodies',
    27:  'Rods & rings'
}

MEAN = [0.08069, 0.05258, 0.05487, 0.08282]
STD = [0.13704, 0.10145, 0.15313, 0.13814]


def generate_folds():
    n_fold = 5
    x = pd.read_csv(os.path.join(PATH, 'train.csv'))
    x = x.sample(frac=1).reset_index(drop=True)
    x.drop('Target', axis=1, inplace=True)
    x['fold'] = (list(range(n_fold))*x.shape[0])[:x.shape[0]]
    print(x.head())
    x.to_csv(os.path.join(DATA, 'folds.csv'), index=False)


def generate_folds_stratified():
    from sklearn.model_selection import StratifiedKFold

    n_fold = 5
    df = pd.read_csv(os.path.join(PATH, 'train.csv'))
    ids = df.Id.values
    labels = df.Target.values
    df['fold'] = 0
    skf = StratifiedKFold(n_splits=n_fold, random_state=123, shuffle=True)
    for fold_no, (train_idx, test_idx) in enumerate(skf.split(ids, labels)):
        df.loc[test_idx, 'fold'] = fold_no
    df.drop('Target', axis=1, inplace=True)
    df.to_csv(os.path.join(DATA, 'folds.csv'), index=False)


def get_test_ids():
    sample_df = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
    return sample_df.Id.values


def get_split(fold, single_class_only=False):
    train_df = pd.read_csv(os.path.join(PATH, 'train.csv'))
    train_ids = [id_ for id_, label_ in train_df[['Id', 'Target']].values if not single_class_only or len(label_.split(' ')) == 1]
        
    folds = pd.read_csv(os.path.join(DATA, 'folds.csv'))
    fold_dict = folds.set_index('Id').to_dict()['fold']

    fold_ids = [train_id for train_id in train_ids if fold_dict[train_id] != fold]
    val_ids = [train_id for train_id in train_ids if fold_dict[train_id] == fold]
    return fold_ids, val_ids


def get_weights_by_class():
    df = pd.read_csv(os.path.join(PATH, 'train.csv'))
    count = Counter()
    freq = {}
    weight = {}

    for row in df.Target.values:
        for label in row.split(' '):
            count[int(label)] += 1

    for klazz in range(NUM_CLASSES):
        freq[klazz] = count[klazz] / len(df)
        weight[klazz] = 1./freq[klazz]

    return freq, weight


def get_sample_weights(train_ids):
    df = pd.read_csv(os.path.join(PATH, 'train.csv'))
    d = df.set_index('Id').to_dict()['Target']
    freq, weight = get_weights_by_class()
    sample_weights = []
    for id_ in train_ids:
        klazzes = [int(x) for x in d[id_].split(' ')]
        # TODO: this is wrong.
        # get better function from notebook
        total_freq = sum([freq[k] for k in klazzes])
        sample_weights.append(1./total_freq)
    return sample_weights


def train_transform_v0():
    return Compose([
        Resize(SIZE, SIZE),
        # RandomCrop(SIZE*2, SIZE*2),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        OneOf([
            RandomBrightness(),
            RandomContrast(),
            RandomGamma(),
            GaussNoise(),
            Blur()
        ]),
        OneOf([
            ElasticTransform(),
            ShiftScaleRotate(
                rotate_limit=45,
                shift_limit=.15,
                scale_limit=.15,
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_REPLICATE),
        ]),
        Resize(SIZE, SIZE),
        Normalize(mean=MEAN, std=STD)
    ])


def train_transform():
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        OneOf([
            RandomBrightness(),
            RandomContrast(),
            RandomGamma(),
            GaussNoise(),
            Blur()
        ]),
        ShiftScaleRotate(
            rotate_limit=45,
            shift_limit=.15,
            scale_limit=.15,
            interpolation=cv2.INTER_CUBIC,
            border_mode=cv2.BORDER_REPLICATE),        
        Resize(SIZE, SIZE),
        Normalize(mean=MEAN, std=STD)
    ])


def val_transform():
    return Compose([
        Resize(SIZE, SIZE),
        Normalize(mean=MEAN, std=STD)
    ])


def make_loader(
        ids,
        shuffle=False,
        num_channels=4,
        transform=None, 
        mode='train',
        batch_size=32, 
        workers=8,
        weighted_sampling=False):
    assert transform is not None

    sampler = None
    filtered_ids = ids
    ds = HPAICDataset(filtered_ids, num_channels=num_channels, transform=transform, mode=mode)

    if weighted_sampling:
        sampler = torch.utils.data.sampler.WeightedRandomSampler(get_sample_weights(filtered_ids), len(filtered_ids))
        shuffle = False  # mutually exclusive with sampler

    return DataLoader(
        dataset=ds,
        shuffle=shuffle,
        num_workers=workers,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True  # torch.cuda.is_available()
    )


class HPAICDataset(Dataset):
    def __init__(self, ids: list, num_channels=3, transform=None, mode='train'):
        self.transform = transform
        self.mode = mode
        self.num_channels = num_channels
        if mode != 'test':
            df = pd.read_csv(os.path.join(PATH, 'train.csv'))
            self.ids_ = df.Id.values
            self.labels = [row.split(' ') for row in df.Target.values]
            self.local_ids = ids
            self.real_idx = dict([(id_, pos) for pos, id_ in enumerate(self.ids_)])
        else:
            self.ids_ = pd.read_csv(os.path.join(PATH, 'sample_submission.csv')).Id.values
            self.local_ids = ids
            self.real_idx = dict([(id_, pos) for pos, id_ in enumerate(self.ids_)])

    def __len__(self):
        return len(self.local_ids)

    def load_image(self, image_id):
        base_path = TEST_PATH if self.mode == 'test' else TRAIN_PATH
        kinds = ['red', 'green', 'blue', 'yellow']
        img = None
        flags = cv2.IMREAD_GRAYSCALE
        for idx, kind in enumerate(kinds):
            path = os.path.join(base_path, f"{image_id}_{kind}.png")
            img_ = cv2.imread(str(path), flags).astype(np.float32) / 255.
            if img is None:
                # we don't know original size, could be 512, or resized during preprocessing to a smaler size.
                original_size = img_.shape[0]
                img = np.zeros((original_size, original_size, len(kinds)), np.float32)
            img[:, :, idx] = img_
        return img

    def load_label(self, image_id):
        y = [0] * (len(label_names))
        for i in self.labels[self.real_idx[image_id]]:
            y[int(i)] = 1
        return np.array(y, np.uint8)

    def __getitem__(self, idx):
        image_id = self.local_ids[idx]
        data = {'image': self.load_image(image_id)}

        augmented = self.transform(**data)

        if self.mode != 'test':
            return img_to_tensor(augmented['image']), torch.from_numpy(self.load_label(image_id)).float()
        else:
            return img_to_tensor(augmented['image']), image_id


def test_ds():
    from collections import Counter
    # generate_folds()
    print('.')
    fold = 0
    fold_ids, val_ids = get_split(fold)
    print('fold ', fold, ' ids: ', len(fold_ids), ', val_ids: ', len(val_ids))

    dl = make_loader(fold_ids, False, num_channels=4, weighted_sampling=True, transform=train_transform(), batch_size=128)
    batch_idx = 0
    for batch_idx, (inputs, targets) in enumerate(dl):
        counter = Counter()
        targets = targets.cpu().numpy()
        for klazz in range(28):
            counter[klazz] += np.sum(targets[:, klazz])
        print(counter.most_common())
        batch_idx += 1
        if batch_idx == 10:
            break


def test_weights():
    fold = 0
    fold_ids, val_ids = get_split(fold)
    weights = get_sample_weights(fold_ids)


if __name__ == '__main__':
    print('.')
    test_ds()


