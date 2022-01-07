import torch.nn as nn
import torch
import sys
import torch.nn.functional as F
sys.path.append('..')

from conditional_batchnorm import CategoricalConditionalBatchNorm2d

class Generator(nn.Module):
	def __init__(self, options, teacher_weight=None, freeze=True):
		super(Generator, self).__init__()
		self.settings = options
		if teacher_weight==None:
			self.label_emb = nn.Embedding(self.settings.n_cls, self.settings.latent_dim)
		else:
			self.label_emb = nn.Embedding.from_pretrained(teacher_weight, freeze=freeze)

		self.embed_normalizer = nn.BatchNorm1d(self.label_emb.weight.T.shape,affine=False,track_running_stats=False)

		if not self.settings.no_DM:
			self.fc_reducer = nn.Linear(in_features=self.label_emb.weight.shape[-1], out_features=self.settings.intermediate_dim)

			self.init_size = self.settings.img_size // 4
			self.l1 = nn.Sequential(nn.Linear(self.settings.intermediate_dim, 128 * self.init_size ** 2))
		else:
			self.init_size = self.settings.img_size // 4
			self.l1 = nn.Sequential(nn.Linear(self.settings.latent_dim, 128 * self.init_size ** 2))

		self.conv_blocks0 = nn.Sequential(
			nn.BatchNorm2d(128),
		)

		self.conv_blocks1 = nn.Sequential(
			nn.Conv2d(128, 128, 3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, inplace=True),
		)
		self.conv_blocks2 = nn.Sequential(
			nn.Conv2d(128, 64, 3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(64, self.settings.channels, 3, stride=1, padding=1),
			nn.Tanh(),
			# nn.BatchNorm2d(self.settings.channels, affine=False)
		)

	def forward(self, z, labels):
		# if linear == None:
		# 	# print(self.label_emb(labels).shape,z.shape)
		# 	gen_input = self.embed_normalizer(torch.add(self.label_emb(labels),self.settings.noise_scale*z).T).T 

		# 	if not self.settings.no_DM:
		# 		gen_input = self.fc_reducer(gen_input)

		# else:
		# 	embed_norm = self.embed_normalizer(torch.add(self.label_emb(labels),self.settings.noise_scale*z).T).T 

		# 	if not self.settings.no_DM:
		# 		gen_input = self.fc_reducer(embed_norm)
		# 	else:
		# 		gen_input = embed_norm

		# 	gen_input = (gen_input * linear.unsqueeze(2)).sum(dim=1)
		gen_input = self.embed_normalizer(torch.add(self.label_emb(labels),self.settings.noise_scale*z).T).T 

		if not self.settings.no_DM:
			gen_input = self.fc_reducer(gen_input)

		out = self.l1(gen_input)
		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
		img = self.conv_blocks0(out)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks1(img)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks2(img)
		return img


class Generator_imagenet(nn.Module):
	def __init__(self, options, teacher_weight=None, freeze=True):
		self.settings = options

		super(Generator_imagenet, self).__init__()

		self.settings = options
		if teacher_weight==None:
			self.label_emb = nn.Embedding(self.settings.nClasses, self.settings.latent_dim)
		else:
			self.label_emb = nn.Embedding.from_pretrained(teacher_weight, freeze=freeze)

		self.embed_normalizer = nn.BatchNorm1d(self.label_emb.weight.T.shape,affine=False,track_running_stats=False)

		if not self.settings.no_DM:
			self.fc_reducer = nn.Linear(in_features=self.label_emb.weight.shape[-1], out_features=self.settings.intermediate_dim)

			self.init_size = self.settings.img_size // 4
			self.l1 = nn.Sequential(nn.Linear(self.settings.intermediate_dim, 128 * self.init_size ** 2))
		else:
			self.init_size = self.settings.img_size // 4
			self.l1 = nn.Sequential(nn.Linear(self.settings.latent_dim, 128 * self.init_size ** 2))

		self.conv_blocks0_0 = CategoricalConditionalBatchNorm2d(1000, 128)

		self.conv_blocks1_0 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
		self.conv_blocks1_1 = CategoricalConditionalBatchNorm2d(1000, 128, 0.8)
		self.conv_blocks1_2 = nn.LeakyReLU(0.2, inplace=True)

		self.conv_blocks2_0 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
		self.conv_blocks2_1 = CategoricalConditionalBatchNorm2d(1000, 64, 0.8)
		self.conv_blocks2_2 = nn.LeakyReLU(0.2, inplace=True)
		self.conv_blocks2_3 = nn.Conv2d(64, self.settings.channels, 3, stride=1, padding=1)
		self.conv_blocks2_4 = nn.Tanh()
		self.conv_blocks2_5 = nn.BatchNorm2d(self.settings.channels, affine=False)

	def forward(self, z, labels, linear=None):
		if linear == None:
			gen_input = self.embed_normalizer(torch.add(self.label_emb(labels),z).T).T
			if not self.settings.no_DM:
				gen_input = self.fc_reducer(gen_input)
		else:
			embed_norm = self.embed_normalizer(torch.add(self.label_emb(labels),z).T).T
			if not self.settings.no_DM:
				gen_input = self.fc_reducer(embed_norm)
			else:
				gen_input = embed_norm
			gen_input = (gen_input * linear.unsqueeze(2)).sum(dim=1)

		out = self.l1(gen_input)
		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
		img = self.conv_blocks0_0(out, labels, linear=linear)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks1_0(img)
		img = self.conv_blocks1_1(img, labels, linear=linear)
		img = self.conv_blocks1_2(img)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks2_0(img)
		img = self.conv_blocks2_1(img, labels, linear=linear)
		img = self.conv_blocks2_2(img)
		img = self.conv_blocks2_3(img)
		img = self.conv_blocks2_4(img)
		img = self.conv_blocks2_5(img)
		return img