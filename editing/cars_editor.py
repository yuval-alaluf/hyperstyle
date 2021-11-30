import torch
import numpy as np
from PIL import Image
import os

from configs.paths_config import edit_paths
from utils.common import tensor2im


class CarsEditor:

    def __init__(self, stylegan_generator):
        self.generator = stylegan_generator
        self.gs = torch.load(edit_paths['cars'])
        # Directions: Pose I, Pose II, Cube, Color, Grass
        self.directions = [(0, 0, 5, 2), (0, 0, 5, -2), (16, 3, 6, 25), (22, 9, 11, -8), (41, 9, 11, -18)]

    def apply_ganspace(self, latents, weights_deltas, input_im, save_dir, noise=None):
        for image_id in latents.keys():
            latent = latents[image_id].to('cuda')
            inputs = [latent]
            for i, (pca_idx, start, end, strength) in enumerate(self.directions):
                delta = self._get_delta(self.gs, latent, pca_idx, strength)
                delta_padded = torch.zeros(latent.shape).to('cuda')
                delta_padded[start:end] += delta.repeat(end - start, 1)
                inputs.append(latent + delta_padded)
            inputs = torch.stack(inputs)
            edited_images = self._latents_to_image(inputs, weights_deltas, noise=noise)
            self._save_coupled_image(input_im, image_id, edited_images, save_dir)

    def _latents_to_image(self, inputs, weights_deltas, noise=None):
        with torch.no_grad():
            images, _ = self.generator([inputs], input_is_latent=True, noise=noise, randomize_noise=False,
                                       weights_deltas=weights_deltas, return_latents=True)
            images = images[:, :, 64:448, :]
        return images

    @staticmethod
    def _get_delta(gs, latent, idx=16, strength=25):
        # gs: ganspace checkpoint loaded, latent: (16, 512) w+
        w_centered = latent - gs['mean'].to('cuda')
        lat_comp = gs['comp'].to('cuda')
        lat_std = gs['std'].to('cuda')
        w_coord = torch.sum(w_centered[0].reshape(-1)*lat_comp[idx].reshape(-1)) / lat_std[idx]
        delta = (strength - w_coord)*lat_comp[idx]*lat_std[idx]
        return delta

    @staticmethod
    def _save_coupled_image(input_im, image_id, edited_images, save_dir):
        res = np.array(input_im)
        for img in edited_images:
            res = np.concatenate([res, tensor2im(img)], axis=1)
        res_im = Image.fromarray(res)
        im_save_path = os.path.join(save_dir, f"{image_id.split('.')[0]}.jpg")
        res_im.save(im_save_path)
