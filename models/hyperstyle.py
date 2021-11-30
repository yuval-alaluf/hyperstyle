import math
import torch
from torch import nn
import copy
from argparse import Namespace

from models.encoders.psp import pSp
from models.stylegan2.model import Generator
from configs.paths_config import model_paths
from models.hypernetworks.hypernetwork import SharedWeightsHyperNetResNet, SharedWeightsHyperNetResNetSeparable
from utils.resnet_mapping import RESNET_MAPPING


class HyperStyle(nn.Module):

    def __init__(self, opts):
        super(HyperStyle, self).__init__()
        self.set_opts(opts)
        self.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
        # Define architecture
        self.hypernet = self.set_hypernet()
        self.decoder = Generator(self.opts.output_size, 512, 8, channel_multiplier=2)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights()
        if self.opts.load_w_encoder:
            self.w_encoder.eval()

    def set_hypernet(self):
        if self.opts.output_size == 1024:
            self.opts.n_hypernet_outputs = 26
        elif self.opts.output_size == 512:
            self.opts.n_hypernet_outputs = 23
        elif self.opts.output_size == 256:
            self.opts.n_hypernet_outputs = 20
        else:
            raise ValueError(f"Invalid Output Size! Support sizes: [1024, 512, 256]!")
        networks = {
            "SharedWeightsHyperNetResNet": SharedWeightsHyperNetResNet(opts=self.opts),
            "SharedWeightsHyperNetResNetSeparable": SharedWeightsHyperNetResNetSeparable(opts=self.opts),
        }
        return networks[self.opts.encoder_type]

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print(f'Loading HyperStyle from checkpoint: {self.opts.checkpoint_path}')
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.hypernet.load_state_dict(self.__get_keys(ckpt, 'hypernet'), strict=True)
            self.decoder.load_state_dict(self.__get_keys(ckpt, 'decoder'), strict=True)
            self.__load_latent_avg(ckpt)
            if self.opts.load_w_encoder:
                self.w_encoder = self.__get_pretrained_w_encoder()
        else:
            hypernet_ckpt = self.__get_hypernet_checkpoint()
            self.hypernet.load_state_dict(hypernet_ckpt, strict=False)
            print(f'Loading decoder weights from pretrained path: {self.opts.stylegan_weights}')
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt['g_ema'], strict=True)
            self.__load_latent_avg(ckpt, repeat=self.n_styles)
            if self.opts.load_w_encoder:
                self.w_encoder = self.__get_pretrained_w_encoder()

    def forward(self, x, resize=True, input_code=False, randomize_noise=True, return_latents=False,
                return_weight_deltas_and_codes=False, weights_deltas=None, y_hat=None, codes=None):

        if input_code:
            codes = x
        else:
            if y_hat is None:
                assert self.opts.load_w_encoder, "Cannot infer latent code when e4e isn't loaded."
                y_hat, codes = self.__get_initial_inversion(x, resize=True)

            # concatenate original input with w-reconstruction or current reconstruction
            x_input = torch.cat([x, y_hat], dim=1)

            # pass through hypernet to get per-layer deltas
            hypernet_outputs = self.hypernet(x_input)
            if weights_deltas is None:
                weights_deltas = hypernet_outputs
            else:
                weights_deltas = [weights_deltas[i] + hypernet_outputs[i] if weights_deltas[i] is not None else None
                                  for i in range(len(hypernet_outputs))]

        input_is_latent = (not input_code)
        images, result_latent = self.decoder([codes],
                                             weights_deltas=weights_deltas,
                                             input_is_latent=input_is_latent,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)

        if resize:
            images = self.face_pool(images)

        if return_latents and return_weight_deltas_and_codes:
            return images, result_latent, weights_deltas, codes, y_hat
        elif return_latents:
            return images, result_latent
        elif return_weight_deltas_and_codes:
            return images, weights_deltas, codes
        else:
            return images

    def set_opts(self, opts):
        self.opts = opts

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None

    def __get_hypernet_checkpoint(self):
        print('Loading hypernet weights from resnet34!')
        hypernet_ckpt = torch.load(model_paths['resnet34'])
        # Transfer the RGB input of the resnet34 network to the first 3 input channels of hypernet
        if self.opts.input_nc != 3:
            shape = hypernet_ckpt['conv1.weight'].shape
            altered_input_layer = torch.randn(shape[0], self.opts.input_nc, shape[2], shape[3], dtype=torch.float32)
            altered_input_layer[:, :3, :, :] = hypernet_ckpt['conv1.weight']
            hypernet_ckpt['conv1.weight'] = altered_input_layer
        mapped_hypernet_ckpt = dict(hypernet_ckpt)
        for p, v in hypernet_ckpt.items():
            for original_name, net_name in RESNET_MAPPING.items():
                if original_name in p:
                    mapped_hypernet_ckpt[p.replace(original_name, net_name)] = v
                    mapped_hypernet_ckpt.pop(p)
        return hypernet_ckpt

    @staticmethod
    def __get_keys(d, name):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
        return d_filt

    def __get_pretrained_w_encoder(self):
        print("Loading pretrained W encoder...")
        opts_w_encoder = vars(copy.deepcopy(self.opts))
        opts_w_encoder['checkpoint_path'] = self.opts.w_encoder_checkpoint_path
        opts_w_encoder['encoder_type'] = self.opts.w_encoder_type
        opts_w_encoder['input_nc'] = 3
        opts_w_encoder = Namespace(**opts_w_encoder)
        w_net = pSp(opts_w_encoder)
        w_net = w_net.encoder
        w_net.eval()
        w_net.cuda()
        return w_net

    def __get_initial_inversion(self, x, resize=True):
        # get initial inversion and reconstruction of batch
        with torch.no_grad():
            return self.__get_w_inversion(x, resize)

    def __get_w_inversion(self, x, resize=True):
        if self.w_encoder.training:
            self.w_encoder.eval()
        codes = self.w_encoder.forward(x)
        if codes.ndim == 2:
            codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
        y_hat, _ = self.decoder([codes],
                                weights_deltas=None,
                                input_is_latent=True,
                                randomize_noise=False,
                                return_latents=False)
        if resize:
            y_hat = self.face_pool(y_hat)
        if "cars" in self.opts.dataset_type:
            y_hat = y_hat[:, :, 32:224, :]
        return y_hat, codes
