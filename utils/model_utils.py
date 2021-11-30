import torch
from argparse import Namespace

import sys
sys.path.extend(['.', '..'])

from models.stylegan2.model import Generator
from models.hyperstyle import HyperStyle
from models.encoders.e4e import e4e


def load_model(checkpoint_path, device='cuda', update_opts=None, is_restyle_encoder=False):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = ckpt['opts']

    opts['checkpoint_path'] = checkpoint_path
    opts['load_w_encoder'] = True

    if update_opts is not None:
        if type(update_opts) == dict:
            opts.update(update_opts)
        else:
            opts.update(vars(update_opts))

    opts = Namespace(**opts)

    if is_restyle_encoder:
        net = e4e(opts)
    else:
        net = HyperStyle(opts)

    net.eval()
    net.to(device)
    return net, opts


def load_generator(checkpoint_path, device='cuda'):
    print(f"Loading generator from checkpoint: {checkpoint_path}")
    generator = Generator(1024, 512, 8, channel_multiplier=2)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    generator.load_state_dict(ckpt['g_ema'])
    generator.eval()
    generator.to(device)
    return generator
