from argparse import ArgumentParser

from configs.paths_config import model_paths


class TestOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        # arguments for inference script
        self.parser.add_argument('--exp_dir', type=str,
                                 help='Path to experiment output directory')
        self.parser.add_argument('--checkpoint_path', default=None, type=str,
                                 help='Path to HyperStyle model checkpoint')
        self.parser.add_argument('--data_path', type=str, default='gt_images',
                                 help='Path to directory of images to evaluate')
        self.parser.add_argument('--resize_outputs', action='store_true',
                                 help='Whether to resize outputs to 256x256 or keep at original output resolution')
        self.parser.add_argument('--test_batch_size', default=2, type=int,
                                 help='Batch size for testing and inference')
        self.parser.add_argument('--test_workers', default=2, type=int,
                                 help='Number of test/inference dataloader workers')
        self.parser.add_argument('--n_images', type=int, default=None,
                                 help='Number of images to output. If None, run on all data')
        self.parser.add_argument('--save_weight_deltas', action='store_true',
                                 help='Whether to save the weight deltas of each image. Note: file weighs about 200MB.')

        # arguments for iterative inference
        self.parser.add_argument('--n_iters_per_batch', default=5, type=int,
                                 help='Number of forward passes per batch during training.')

        # arguments for loading pre-trained encoder
        self.parser.add_argument('--load_w_encoder', action='store_true', help='Whether to load the w e4e encoder.')
        self.parser.add_argument('--w_encoder_checkpoint_path', default=model_paths["faces_w_encoder"], type=str,
                                 help='Path to pre-trained W-encoder.')
        self.parser.add_argument('--w_encoder_type', default='WEncoder',
                                 help='Encoder type for the encoder used to get the initial inversion')

        # arguments for editing scripts 
        self.parser.add_argument('--edit_directions', default='age,smile,pose', help='which edit directions top perform')
        self.parser.add_argument('--factor_range', type=int, default=5, help='max range for interfacegan edits.')

        # arguments for domain adaptation
        self.parser.add_argument('--restyle_checkpoint_path', default=model_paths["restyle_e4e_ffhq"], type=str,
                                 help='ReStyle e4e checkpoint path used for domain adaptation')
        self.parser.add_argument('--restyle_n_iterations', default=2, type=int,
                                 help='Number of forward passes per batch for ReStyle-e4e inference.')
        self.parser.add_argument('--finetuned_generator_checkpoint_path', type=str, default=model_paths["stylegan_pixar"],
                                 help='Path to fine-tuned generator checkpoint used for domain adaptation.')

    def parse(self):
        opts = self.parser.parse_args()
        return opts
