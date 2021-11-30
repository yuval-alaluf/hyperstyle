import argparse
import os
import pickle
import torch
import numpy as np
import torchvision

import sys
sys.path.append(".")
sys.path.append("..")

from configs.paths_config import edit_paths, model_paths
from editing.styleclip.global_direction import StyleCLIPGlobalDirection
from editing.styleclip.model import Generator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="./experiment",
                        help="Path to inference results with `latents.npy` saved here (obtained with inference.py).")
    parser.add_argument("--weight_deltas_path", type=str, default="./weight_deltas",
                        help="Root path holding all weight deltas (obtained by running inference.py).")
    parser.add_argument('--n_images', type=int, default=None,
                        help="Maximum number of images to edit. If None, edit all images.")
    parser.add_argument("--neutral_text", type=str, default="face with hair")
    parser.add_argument("--target_text", type=str, default="face with long hair")
    parser.add_argument("--stylegan_weights", type=str, default=model_paths["stylegan_ffhq"])
    parser.add_argument("--stylegan_size", type=int, default=1024)
    parser.add_argument("--stylegan_truncation", type=int, default=1.)
    parser.add_argument("--stylegan_truncation_mean", type=int, default=4096)
    parser.add_argument("--beta_min", type=float, default=0.11)
    parser.add_argument("--beta_max", type=float, default=0.16)
    parser.add_argument("--num_betas", type=int, default=5)
    parser.add_argument("--alpha_min", type=float, default=-5)
    parser.add_argument("--alpha_max", type=float, default=5)
    parser.add_argument("--num_alphas", type=int, default=11)
    parser.add_argument("--delta_i_c", type=str, default=edit_paths["styleclip"]["delta_i_c"],
                        help="path to file containing delta_i_c")
    parser.add_argument("--s_statistics", type=str, default=edit_paths["styleclip"]["s_statistics"],
                        help="path to file containing s statistics")
    parser.add_argument("--text_prompt_templates", default=edit_paths["styleclip"]["templates"])
    args = parser.parse_args()
    return args


def load_direction_calculator(args):
    delta_i_c = torch.from_numpy(np.load(args.delta_i_c)).float().cuda()
    with open(args.s_statistics, "rb") as channels_statistics:
        _, s_std = pickle.load(channels_statistics)
        s_std = [torch.from_numpy(s_i).float().cuda() for s_i in s_std]
    with open(args.text_prompt_templates, "r") as templates:
        text_prompt_templates = templates.readlines()
    global_direction_calculator = StyleCLIPGlobalDirection(delta_i_c, s_std, text_prompt_templates)
    return global_direction_calculator


def load_stylegan_generator(args):
    stylegan_model = Generator(args.stylegan_size, 512, 8, channel_multiplier=2).cuda()
    checkpoint = torch.load(args.stylegan_weights)
    stylegan_model.load_state_dict(checkpoint['g_ema'])
    return stylegan_model


def run():
    args = parse_args()
    stylegan_model = load_stylegan_generator(args)
    global_direction_calculator = load_direction_calculator(args)
    # load latents obtained via inference
    latents = np.load(os.path.join(args.exp_dir, 'latents.npy'), allow_pickle=True).item()
    # prepare output directory
    args.output_path = os.path.join(args.exp_dir, "styleclip_edits", f"{args.neutral_text}_to_{args.target_text}")
    os.makedirs(args.output_path, exist_ok=True)
    # edit all images
    for idx, (image_name, latent) in enumerate(latents.items()):
        if args.n_images is not None and idx >= args.n_images:
            break
        edit_image(image_name, latent, stylegan_model, global_direction_calculator, args)


def edit_image(image_name, latent, stylegan_model, global_direction_calculator, args):
    print(f'Editing {image_name}')

    latent_code = torch.from_numpy(latent).cuda()
    truncation = 1
    mean_latent = None
    input_is_latent = True
    latent_code_i = latent_code.unsqueeze(0)

    weight_deltas = np.load(os.path.join(args.weight_deltas_path, image_name.split(".")[0] + ".npy"), allow_pickle=True)
    weight_deltas = [torch.from_numpy(w).cuda() if w is not None else None for w in weight_deltas]

    with torch.no_grad():

        source_im, _, latent_code_s = stylegan_model([latent_code_i],
                                                     input_is_latent=input_is_latent,
                                                     randomize_noise=False,
                                                     return_latents=True,
                                                     truncation=truncation,
                                                     truncation_latent=mean_latent,
                                                     weights_deltas=weight_deltas)

    alphas = np.linspace(args.alpha_min, args.alpha_max, args.num_alphas)
    betas = np.linspace(args.beta_min, args.beta_max, args.num_betas)
    results = []
    for beta in betas:
        direction = global_direction_calculator.get_delta_s(args.neutral_text, args.target_text, beta)
        edited_latent_code_s = [[s_i + alpha * b_i for s_i, b_i in zip(latent_code_s, direction)] for alpha in alphas]
        edited_latent_code_s = [torch.cat([edited_latent_code_s[i][j] for i in range(args.num_alphas)])
                                for j in range(len(edited_latent_code_s[0]))]
        for b in range(0, edited_latent_code_s[0].shape[0]):
            edited_latent_code_s_batch = [s_i[b:b + 1] for s_i in edited_latent_code_s]
            with torch.no_grad():
                edited_image, _, _ = stylegan_model([edited_latent_code_s_batch],
                                                    input_is_stylespace=True,
                                                    randomize_noise=False,
                                                    return_latents=True,
                                                    weights_deltas=weight_deltas)
                results.append(edited_image)

    results = torch.cat(results)
    torchvision.utils.save_image(results, f"{args.output_path}/{image_name.split('.')[0]}.jpg",
                                 normalize=True, range=(-1, 1), padding=0, nrow=args.num_alphas)


if __name__ == "__main__":
    run()
