"""
Reference:
https://github.com/FedML-AI/FedML
"""
import logging
import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from logging import handlers
from datasets import CIFAR10_truncated
from datasets import CIFAR100_truncated
from datasets import GetLoader

import matplotlib.pyplot as plt

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def read_data_distribution(filename='./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'):
    distribution = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0]:
                tmp = x.split(':')
                if '{' == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(',', ''))
    return distribution


def read_net_dataidx_map(filename='./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'):
    net_dataidx_map = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0] and ']' != x[0]:
                tmp = x.split(':')
                if '[' == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(',')
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


class Cutout(object): 
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(): 
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124] 
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768] 

    # CIFAR_MEAN = [0.50707516, 0.48654887, 0.44091784] #cifar100
    # CIFAR_STD = [0.26733429, 0.25643846, 0.27615047]


    train_transform = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(), 
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD), 
    ])

    train_transform.transforms.append(Cutout(16)) 
    
    valid_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD), 
    ])


    return train_transform , valid_transform


def partition_data_dataset(X_train,y_train, n_nets, alpha): 
    
    min_size = 0 
    K = 10
    # k =20 CIFAR-100
    N = y_train.shape[0]
    logging.info("N = " + str(N)) 
    net_dataidx_map = {}

    while min_size < 10: 
        idx_batch = [[] for _ in range(n_nets)] 
        for k in range(K):
            idx_k = np.where(y_train == k)[0] 
            np.random.seed(k) 
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets)) 
            np.random.shuffle(idx_k) 
            proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)]) 
            proportions = proportions / proportions.sum() 
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1] 
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))] 
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j]) 
        net_dataidx_map[j] = idx_batch[j] 

    return net_dataidx_map

def partition_data_mean(X_train,y_train, n_nets):
        net_dataidx_map = {}
        K = 3
        idx_batch = [[] for _ in range(n_nets)] 
        for k in range(K):
            idx_k = np.where(y_train == k)[0] 
            np.random.seed(k) 
            np.random.shuffle(idx_k) 
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.array_split(idx_k, n_nets))]
        for j in range(n_nets):
            np.random.shuffle(idx_batch[j]) 
            net_dataidx_map[j] = idx_batch[j] 
        return net_dataidx_map

def partition_data(train_dataset,test_dataset ,datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    X_train=train_dataset.data
    y_train=train_dataset.target
    X_test=test_dataset.data
    y_test=test_dataset.target
    n_train = X_train.shape[0] 
    print(n_train)
    n_test = X_test.shape[0]

    if partition == "homo":
        total_num = n_train
        test_total_num= n_test
        idxs = np.random.permutation(total_num) 
        idxs_test= np.random.permutation(test_total_num) 
        batch_idxs = np.array_split(idxs, n_nets) 
        batch_idxs_test = np.array_split(idxs_test, n_nets) 
        net_dataidx_map_train = {i: batch_idxs[i] for i in range(n_nets)} 
        net_dataidx_map_test={i: batch_idxs_test[i] for i in range(n_nets)} 
    elif partition == "hetero": 
        net_dataidx_map_train=partition_data_dataset(X_train,y_train,n_nets,alpha) 
        net_dataidx_map_test=partition_data_dataset(X_test,y_test,n_nets,alpha) 
    elif partition == "mean":
        net_dataidx_map_train=partition_data_mean(X_train,y_train,n_nets)
        net_dataidx_map_test=partition_data_mean(X_test,y_test,n_nets)
    else:
        raise Exception("partition arg error")

    return X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test 


def get_dataloader(train_dataset, test_dataset, datadir, train_bs, test_bs, dataidxs_train=None,dataidxs_test=None):
    return get_dataloader_CIFAR10(train_dataset, test_dataset, datadir, train_bs, test_bs, dataidxs_train,dataidxs_test)


def get_dataloader_CIFAR10(train_dataset, test_dataset, datadir, train_bs, test_bs, dataidxs_train=None,dataidxs_test=None):

    train_transform,test_transform = _data_transforms_cifar10()

    train_ds = GetLoader(train_dataset,dataidxs=dataidxs_train,transform=train_transform)
    test_ds = GetLoader(test_dataset,dataidxs=dataidxs_test,transform=test_transform) 

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=False, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }
    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str) 
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh) 
        self.logger.addHandler(th)
log=Logger('client_local_sample_number',level='info')

def test_train_dataset():
    train_transform, test_transform = _data_transforms_cifar10()

    #cifar10 train data
    all_train_data = CIFAR10_truncated(root='./data',train=True,download=False,transform=train_transform) #(50000,32,32,3)

    numbers = list(range(50000))  
    client_train_index=numbers
    client_train_data=GetLoader(all_train_data,dataidxs=client_train_index)

    # cifar10 test data
    client_test_data = CIFAR10_truncated(root='./data',train=False,download=False,transform=train_transform) #(10000,32,32,3)

    return client_train_data,client_test_data

def load_partition_data_cifar10(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    to_pil = ToPILImage()
    train_data,test_data=test_train_dataset()
    X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test = partition_data(train_data,
                                                                                             test_data,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)

    class_num_train = len(np.unique(y_train)) 
    class_num_test = len(np.unique(y_test)) 

    train_data_num = sum([len(net_dataidx_map_train[r]) for r in range(client_number)]) 
    test_data_num = sum([len(net_dataidx_map_test[r]) for r in range(client_number)]) 

    train_data_global, test_data_global = get_dataloader(train_data,test_data,data_dir, batch_size, batch_size)

    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))

    data_local_num_dict_train = dict()
    data_local_num_dict_test = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()


    for client_idx in range(client_number):
        dataidxs_train = net_dataidx_map_train[client_idx]
        dataidxs_test =  net_dataidx_map_test[client_idx] 

        local_data_num_train = len(dataidxs_train) 
        local_data_num_test = len(dataidxs_test) 

        data_local_num_dict_train[client_idx] = local_data_num_train 
        data_local_num_dict_test[client_idx] = local_data_num_test

        log.logger.info("client_idx = %d, train_local_sample_number = %d" % (client_idx, local_data_num_train))
        log.logger.info("client_idx = %d, test_local_sample_number = %d" % (client_idx, local_data_num_test))

        train_data_local, test_data_local= get_dataloader(train_data, test_data, data_dir, batch_size, batch_size, dataidxs_train,dataidxs_test)

        log.logger.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
             client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local 
        test_data_local_dict[client_idx] = test_data_local 


    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    return train_data_num, test_data_num, train_data_global, test_data_global,\
           data_local_num_dict_train, data_local_num_dict_test,train_data_local_dict, test_data_local_dict, class_num_train,class_num_test
