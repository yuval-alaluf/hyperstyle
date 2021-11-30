import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils


class LatentsImagesDataset(Dataset):

	def __init__(self, source_root, target_root, latents_path, opts, target_transform=None, source_transform=None):
		# path to inversions directory
		self.source_root = source_root 
		# path to original dataset
		self.target_paths = sorted(data_utils.make_dataset(target_root)) 
		# path to latents corresponding to inversions
		# this should be a dictionary mapping image name to the image's latent code
		self.latents = torch.load(latents_path, map_location='cpu')  
		self.latents.requires_grad = False
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts

	def __len__(self):
		return len(self.target_paths)

	def __getitem__(self, index):
		from_path = os.path.join(self.source_root, f'{index+1:05d}.png')
		to_path = self.target_paths[index]

		from_im = Image.open(from_path).convert('RGB')
		to_im = Image.open(to_path).convert('RGB')

		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		latent = self.latents[os.path.basename(from_path)]
		if latent.ndim == 1:
			latent = latent.repeat(18, 1)

		return from_im, to_im, latent
