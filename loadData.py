import argparse
import numpy as np
import torch
from data.HelpLoad import unpickle
from torchvision import transforms
from torch.utils.data import dataset, dataloader


class LoadDataset(dataset.Dataset):
    def __init__(self, data_file_list, num_classes, data_dir='data/', transform=None):
        self.x = []
        self.y = []
        # y = []
        for file in data_file_list:
            data_dict = unpickle(data_dir+file)
            self.x.extend(data_dict[b'data'])
            self.y.extend(data_dict[b'labels'])

        # for i in y:
        #     a = torch.zeros(num_classes)
        #     a[i] = 1
        #     self.y.append(a)

        self.length = len(self.y)
        self.trans = transform

    def __getitem__(self, i):
        arr = np.array(self.x[i]).reshape(3,32,32).transpose(1,2,0)
        if self.trans:
            arr = self.trans(arr)
        return arr, self.y[i]

    def __len__(self):
        return self.length


def get_test_trans():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(28),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])


def get_train_trans():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(28),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_dataloader(mode, data_file_list, opt: argparse.Namespace):
    if mode == 'train':
        dataset = LoadDataset(
            data_file_list=data_file_list,
            num_classes=opt.num_classes,
            transform=get_train_trans()
        )
        return dataloader.DataLoader(
            dataset=dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.tr_dl_num_worker
        )#, dataset.label_statistics
    elif mode == 'test':
        return dataloader.DataLoader(
            dataset=LoadDataset(
                data_file_list=data_file_list,
                num_classes=opt.num_classes,
                transform=get_test_trans()
            ),
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.te_dl_num_worker
        )
    else:
        raise ValueError('Unknown mode: %s' % mode)
