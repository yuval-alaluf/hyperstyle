import os
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import sys
sys.path.extend([".", ".."])

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from options.test_options import TestOptions
from utils.common import tensor2im
from utils.domain_adaptation_utils import run_domain_adaptation
from utils.model_utils import load_model, load_generator


def run():
    test_opts = TestOptions().parse()

    out_path_results = os.path.join(test_opts.exp_dir, 'domain_adaptation_results')
    out_path_coupled = os.path.join(test_opts.exp_dir, 'domain_adaptation_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    net, opts = load_model(test_opts.checkpoint_path, update_opts=test_opts)

    restyle_e4e, restyle_opts = load_model(test_opts.restyle_checkpoint_path,
                                           update_opts={"resize_outputs": test_opts.resize_outputs,
                                                        "n_iters_per_batch": test_opts.restyle_n_iterations},
                                           is_restyle_encoder=True)
    finetuned_generator = load_generator(test_opts.finetuned_generator_checkpoint_path)

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

    global_i = 0
    for input_batch in tqdm(dataloader):

        if global_i >= opts.n_images:
            break

        with torch.no_grad():
            input_cuda = input_batch.cuda().float()
            result_batch, _ = run_domain_adaptation(input_cuda, net, opts, finetuned_generator,
                                                    restyle_e4e, restyle_opts)

        resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
        for i in range(input_batch.shape[0]):

            im_path = dataset.paths[global_i]

            curr_result = tensor2im(result_batch[i])
            input_im = tensor2im(input_batch[i])

            res_save_path = os.path.join(out_path_results, os.path.basename(im_path))
            curr_result.resize(resize_amount).save(res_save_path)

            coupled_save_path = os.path.join(out_path_coupled, os.path.basename(im_path))
            res = np.concatenate([np.array(input_im.resize(resize_amount)), np.array(curr_result.resize(resize_amount))],
                                 axis=1)
            Image.fromarray(res).save(coupled_save_path)
            global_i += 1


if __name__ == '__main__':
    run()
