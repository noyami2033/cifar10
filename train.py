import argparse
import torch
import os

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model import Network
from dataset import DataSet
from dataset import Normalize, ToTensor
import utils

def test_accuracy(estimate, target):
	_, indices = torch.max(estimate, 1)
	true_num = (indices == target).sum()
	total_num = target.size(0)
	acc = true_num/total_num
	return true_num, total_num


parser = argparse.ArgumentParser(description='training setting')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.1, help='Learning Rate.')
parser.add_argument('--data_dir', type=str, default='cifar-10/', help='data directry path')
parser.add_argument('--train_annotations', type=str, default='cifar-10/train_annotations.txt', help='annotation file path')
parser.add_argument('--test_annotations', type=str, default='cifar-10/test_annotations.txt', help='annotation file path')
parser.add_argument('--show_every', type=int, default=50, help='visualize image every # iteration')
opt = parser.parse_args()

model = Network()
if torch.cuda.is_available():
	model.cuda()
optimizer = torch.optim.Adadelta(model.parameters(), lr=opt.lr)
CEloss=torch.nn.CrossEntropyLoss()

train_dataset = DataSet(data_dir=opt.data_dir, annotation_path=opt.train_annotations, transform=transforms.Compose([Normalize(), ToTensor()]))
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, drop_last=True, shuffle=True)
test_dataset = DataSet(data_dir=opt.data_dir, annotation_path=opt.test_annotations, transform=transforms.Compose([Normalize(), ToTensor()]))
test_loader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size, drop_last=True, shuffle=True)

for epoch in range(opt.num_epochs):
	sum_true_num = 0
	sum_total_num = 0
	sum_loss = 0
	for i, sample in enumerate(train_loader):
		iteration = epoch * len(train_loader) + i
		optimizer.zero_grad()

		images = utils.get_variable(sample["image"])
		labels = utils.get_variable(sample["label"]).squeeze(1)

		outputs = model(images)
		loss = CEloss(outputs, labels)
		loss.backward()
		optimizer.step()

		true_num, total_num = test_accuracy(outputs, labels)
		sum_true_num += true_num
		sum_total_num += total_num
		sum_loss += loss

		if iteration%opt.show_every == 0:
			print("epoch: {}, iteration: {}, loss: {:.6f}, accuracy: {:.6f}".format(epoch, iteration, sum_loss/sum_total_num, sum_true_num/sum_total_num))
			sum_true_num = 0
			sum_total_num = 0
			sum_loss = 0

	model.eval()
	sum_true_num = 0
	sum_total_num = 0
	for i, sample in enumerate(test_loader):
		images = utils.get_variable(sample["image"])
		labels = utils.get_variable(sample["label"]).squeeze(1)
		outputs = model(images)
		true_num, total_num = test_accuracy(outputs, labels)
		sum_true_num += true_num
		sum_total_num += total_num

	print("---------------")
	print("epoch: {}, accuracy: {:.6f}".format(epoch, train_acc))
	print("---------------")
	torch.save(model.state_dict(), "model.pth")
	model.train()
	