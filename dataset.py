from torch.utils.data import Dataset
import torch
import cv2
import numpy as np

class DataSet(Dataset):
	def __init__(self, data_dir, annotation_path, transform=None):
		self.image_paths = []
		self.labels = []
		self.data_dir = data_dir
		self.transform = transform
		with open(annotation_path, 'r') as f:
			for line in f:
				img_path, img_label = line.split(' ')
				self.image_paths.append(img_path)
				self.labels.append(img_label)

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		img_path = self.data_dir + self.image_paths[idx]
		image = cv2.imread(img_path)

		label = np.array([self.labels[idx]], dtype=np.int8)
		image = image.transpose((2, 0, 1))

		sample = {"image":image, "label":label}

		if self.transform:
			sample = self.transform(sample)

		return sample

class Normalize(object):
	def __call__(self, sample):
		images, labels = sample['image'], sample['label']
		images = (images - np.mean(images))/np.std(images)
		return {'image': images,
				'label': labels}

class ToTensor(object):
	def __call__(self, sample):
		images, labels = sample['image'], sample['label']
		return {'image': torch.from_numpy(images).float(),
				'label': torch.from_numpy(labels).long()}
