import os
from tqdm import tqdm
import time
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from editing.cars_editor import CarsEditor
from options.test_options import TestOptions
from utils.inference_utils import run_inversion
from utils.model_utils import load_model
from utils.common import tensor2im


def run():
    test_opts = TestOptions().parse()

    out_path_coupled = os.path.join(test_opts.exp_dir, 'editing_coupled')
    os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    net, opts = load_model(test_opts.checkpoint_path, update_opts=test_opts)

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=False)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    latent_editor = CarsEditor(net.decoder)

    global_i = 0
    global_time = []
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break
        with torch.no_grad():
            input_cuda = input_batch.cuda().float()
            tic = time.time()
            y_hat, batch_latents, weights_deltas, codes = run_inversion(input_cuda, net, opts)
            toc = time.time()
            global_time.append(toc - tic)

        for i in range(input_batch.shape[0]):

            im_path = dataset.paths[global_i]
            input_im = tensor2im(input_batch[i]).resize((512, 384))

            sample_deltas = [d[i] if d is not None else None for d in weights_deltas]
            latents = {os.path.basename(im_path): batch_latents[i]}
            latent_editor.apply_ganspace(latents=latents,
                                         weights_deltas=sample_deltas,
                                         input_im=input_im,
                                         save_dir=out_path_coupled)
            global_i += 1


if __name__ == '__main__':
    run()
