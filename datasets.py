"""
Reference:
https://github.com/FedML-AI/FedML
"""
import logging
from autoencoder_pretrained import create_autoencoder
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import pandas as pd
from scipy.io.arff import loadarff

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')




class GetLoader(data.Dataset): 
    def __init__(self, dataset, dataidxs=None, transform=None, target_transform=None):

        self.dataset = dataset
        self.dataidxs = dataidxs
        self.transform = transform
        self.target_transform = target_transform
        self.data, self.target = self.__build_truncated_dataset__() 

    def __build_truncated_dataset__(self):
        dataobj = self.dataset 

        data = dataobj.data
        target = dataobj.target

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
    

class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__() 

    def __build_truncated_dataset__(self):
        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download) 

        if self.train: 
            # print("train member of the class: {}".format(self.train))
            # data = cifar_dataobj.train_data
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    
class CIFAR100_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__() 

    def __build_truncated_dataset__(self):
        print("download = " + str(self.download)) 
        cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download) 

        if self.train: 
            coarse_fine_dict={
                0:[4,30,55,72,95],
                1:[1,32,67,73,91],
                2:[54,62,70,82,92],
                3:[9,10,16,28,61],
                4:[0,51,53,57,83],
                5:[22,39,40,86,87],
                6:[5,20,25,84,94],
                7:[6,7,14,18,24],
                8:[3,42,43,88,97],
                9:[12,17,37,68,76],
                10:[23, 33, 49, 60, 71],
                11:[15, 19, 21, 31, 38],
                12:[34, 63, 64, 66, 75],
                13:[26, 45, 77, 79, 99],
                14:[2, 11, 35, 46, 98],
                15:[27, 29,44, 78, 93],
                16:[36,50,65, 74,80],
                17:[47, 52,56,59,96],
                18:[8,13,48,58,90],
                19:[41,69,81,85,89]
            }
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
            for idx,label in enumerate(target):
                for key,values in coarse_fine_dict.items():
                    if label in values:
                        target[idx] = key
        else:
            coarse_fine_dict={
                0:[4,30,55,72,95],
                1:[1,32,67,73,91],
                2:[54,62,70,82,92],
                3:[9,10,16,28,61],
                4:[0,51,53,57,83],
                5:[22,39,40,86,87],
                6:[5,20,25,84,94],
                7:[6,7,14,18,24],
                8:[3,42,43,88,97],
                9:[12,17,37,68,76],
                10:[23, 33, 49, 60, 71],
                11:[15, 19, 21, 31, 38],
                12:[34, 63, 64, 66, 75],
                13:[26, 45, 77, 79, 99],
                14:[2, 11, 35, 46, 98],
                15:[27, 29,44, 78, 93],
                16:[36,50,65, 74,80],
                17:[47, 52,56,59,96],
                18:[8,13,48,58,90],
                19:[41,69,81,85,89]
            }
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
            for idx,label in enumerate(target):
                for key,values in coarse_fine_dict.items():
                    if label in values:
                        target[idx] = key

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)