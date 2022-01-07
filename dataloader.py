import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as T
import socket
import struct

__all__ = ["DataLoader", "PartDataLoader"]

def get_data_folder(opt):
    """
    return server-dependent path to store the data
    """
    hostname = socket.gethostname()
    if hostname.startswith('visiongpu'):
        data_folder = '/data/vision/phillipi/rep-learn/datasets'
    elif hostname.startswith('yonglong-home'):
        data_folder = '/home/yonglong/Data/data'
    else:
        data_folder = opt.df_folder

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    test_folder = '/data/lijingru/' + opt.dataset + '/'

    return data_folder, test_folder

class ImageLoader(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        class_list = os.listdir(root)
        datasets = []
        for cla in class_list:
            cla_path = os.path.join(root, cla)
            files = os.listdir(cla_path)
            for f in files:
                file_path = os.path.join(cla_path, f)
                if os.path.isfile(file_path):
                    datasets.append((file_path, [float(cla)]))

        self.root = root
        self.datasets = datasets
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        # frames = []
        file_path, label = self.datasets[idx]
        noise = torch.load(file_path, map_location=torch.device('cpu'))
        return noise, torch.Tensor(label)

    def __len__(self):
        return len(self.datasets)


class DataLoader(object):
	"""
	data loader for CV data sets
	"""
	
	def __init__(self, dataset, batch_size, n_threads=4,
	             ten_crop=False, data_path='/home/dataset/', logger=None):
		"""
		create data loader for specific data set
		:params n_treads: number of threads to load data, default: 4
		:params ten_crop: use ten crop for testing, default: False
		:params data_path: path to data set, default: /home/dataset/
		"""
		self.dataset = dataset
		self.batch_size = batch_size
		self.n_threads = n_threads
		self.ten_crop = ten_crop
		self.data_path = data_path
		self.logger = logger
		self.dataset_root = data_path
		
		self.logger.info("|===>Creating data loader for " + self.dataset)
		
		if self.dataset in ["cifar100","cifar10"]:
			self.train_loader, self.test_loader = self.cifar(
				dataset=self.dataset)
		
		elif self.dataset in ["imagenet"]:
			self.train_loader, self.test_loader = self.imagenet(
				dataset=self.dataset)
		else:
			assert False, "invalid data set"
	
	def getloader(self):
		"""
		get train_loader and test_loader
		"""
		return self.train_loader, self.test_loader

	def imagenet(self, dataset="imagenet"):

		traindir = os.path.join(self.data_path, "train")
		testdir = os.path.join(self.data_path, "val")

		normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
										 std=[0.229, 0.224, 0.225])

		train_loader = None

		test_transform = T.Compose([
			T.Resize(256),
			T.CenterCrop(224),
			T.ToTensor(),
			normalize
		])

		test_loader = torch.utils.data.DataLoader(
			dsets.ImageFolder(testdir, test_transform),
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.n_threads,
			pin_memory=False)
		return train_loader, test_loader

	def cifar(self, dataset="cifar100"):
		"""
		dataset: cifar
		"""
		if dataset == "cifar10":
			norm_mean = [0.49139968, 0.48215827, 0.44653124]
			norm_std = [0.24703233, 0.24348505, 0.26158768]
		elif dataset == "cifar100":
			norm_mean = [0.50705882, 0.48666667, 0.44078431]
			norm_std = [0.26745098, 0.25568627, 0.27607843]
		
		else:
			assert False, "Invalid cifar dataset"

		test_data_root = self.dataset_root

		test_transform = T.Compose([
			T.ToTensor(),
			T.Normalize(norm_mean, norm_std)])

		if self.dataset == "cifar10":
			test_dataset = dsets.CIFAR10(root=test_data_root,
			                             train=False,
			                             transform=test_transform,download=True)
		elif self.dataset == "cifar100":
			test_dataset = dsets.CIFAR100(root=test_data_root,
			                              train=False,
			                              transform=test_transform,
			                              download=True)
		else:
			assert False, "invalid data set"

		test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
												  batch_size=200,
												  shuffle=False,
												  pin_memory=True,
												  num_workers=self.n_threads)
		return None, test_loader


class CIFAR100Gen(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, return_target=False):
        self.root = root
        self.files = os.listdir(root)
        self.transform = transform
        self.target_transform = target_transform
        self.return_target = return_target
    
    def __getitem__(self, idx):
        f = os.path.join(self.root, self.files[idx])

        img = Image.open(f)
        if self.transform is not None:
            img = self.transform(img)
        
        if self.return_target:
            target = int(self.files[idx].split('_')[2])
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target
        else:
            return img
    
    def __len__(self):
        return len(self.files)


def get_train_loader(opt):
	data_folder, test_folder = get_data_folder(opt)
	dataset = opt.dataset
	if dataset == "cifar10":
		norm_mean = [0.49139968, 0.48215827, 0.44653124]
		norm_std = [0.24703233, 0.24348505, 0.26158768]
	elif dataset == "cifar100":
		norm_mean = [0.50705882, 0.48666667, 0.44078431]
		norm_std = [0.26745098, 0.25568627, 0.27607843]
	train_transform = T.Compose([
		T.ToTensor(),
		T.Normalize(norm_mean, norm_std)])
	train_set = CIFAR100Gen(root=data_folder, transform=train_transform, return_target=True)
	train_loader = data.DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
	return train_loader