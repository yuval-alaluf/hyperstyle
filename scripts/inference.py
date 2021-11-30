import os

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import sys
sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im
from utils.inference_utils import run_inversion
from utils.model_utils import load_model
from options.test_options import TestOptions


def run():
    test_opts = TestOptions().parse()

    out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
    out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

    os.makedirs(out_path_results, exist_ok=True)
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

    if "cars" in opts.dataset_type:
        resize_amount = (256, 192) if opts.resize_outputs else (512, 384)
    else:
        resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)

    global_i = 0
    global_time = []
    all_latents = {}
    for input_batch in tqdm(dataloader):

        if global_i >= opts.n_images:
            break

        with torch.no_grad():
            input_cuda = input_batch.cuda().float()
            tic = time.time()
            result_batch, result_latents, result_deltas = run_inversion(input_cuda, net, opts,
                                                                        return_intermediate_results=True)
            toc = time.time()
            global_time.append(toc - tic)

        for i in range(input_batch.shape[0]):
            results = [tensor2im(result_batch[i][iter_idx]) for iter_idx in range(opts.n_iters_per_batch)]
            im_path = dataset.paths[global_i]

            input_im = tensor2im(input_batch[i])
            res = np.array(input_im.resize(resize_amount))
            for idx, result in enumerate(results):
                res = np.concatenate([res, np.array(result.resize(resize_amount))], axis=1)
                # save individual outputs
                save_dir = os.path.join(out_path_results, str(idx))
                os.makedirs(save_dir, exist_ok=True)
                result.resize(resize_amount).save(os.path.join(save_dir, os.path.basename(im_path)))

            # save coupled image with side-by-side results
            Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

            all_latents[os.path.basename(im_path)] = result_latents[i][0]

            if opts.save_weight_deltas:
                weight_deltas_dir = os.path.join(test_opts.exp_dir, "weight_deltas")
                os.makedirs(weight_deltas_dir, exist_ok=True)
                np.save(os.path.join(weight_deltas_dir, os.path.basename(im_path).split('.')[0] + ".npy"),
                        result_deltas[i][-1])

            global_i += 1

    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str)

    # save all latents as npy file
    np.save(os.path.join(test_opts.exp_dir, 'latents.npy'), all_latents)


if __name__ == '__main__':
    run()
