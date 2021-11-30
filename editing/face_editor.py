import torch

from configs.paths_config import edit_paths
from utils.common import tensor2im


class FaceEditor:

    def __init__(self, stylegan_generator):
        self.generator = stylegan_generator
        self.interfacegan_directions = {
            'age': torch.load(edit_paths['age']).cuda(),
            'smile': torch.load(edit_paths['smile']).cuda(),
            'pose': torch.load(edit_paths['pose']).cuda()
        }

    def apply_interfacegan(self, latents, weights_deltas, direction, factor=1, factor_range=None):
        edit_latents = []
        direction = self.interfacegan_directions[direction]
        if factor_range is not None:  # Apply a range of editing factors. for example, (-5, 5)
            for f in range(*factor_range):
                edit_latent = latents + f * direction
                edit_latents.append(edit_latent)
            edit_latents = torch.stack(edit_latents).transpose(0,1)
        else:
            edit_latents = latents + factor * direction
        return self._latents_to_image(edit_latents, weights_deltas)

    def _latents_to_image(self, all_latents, weights_deltas):
        sample_results = {}
        with torch.no_grad():
            for idx, sample_latents in enumerate(all_latents):
                sample_deltas = [d[idx] if d is not None else None for d in weights_deltas]
                images, _ = self.generator([sample_latents],
                                           weights_deltas=sample_deltas,
                                           randomize_noise=False,
                                           input_is_latent=True)
                sample_results[idx] = [tensor2im(image) for image in images]
        return sample_results
