from argparse import ArgumentParser

from configs.paths_config import model_paths


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        # general setup
        self.parser.add_argument('--exp_dir', type=str,
                                 help='Path to experiment output directory')
        self.parser.add_argument('--dataset_type', default='ffhq_encode', type=str,
                                 help='Type of dataset/experiment to run')
        self.parser.add_argument('--encoder_type', default='HyperNet', type=str,
                                 help='Which encoder to use')
        self.parser.add_argument('--input_nc', default=6, type=int,
                                 help='Number of input image channels to the HyperStyle network. Should be set to 6.')
        self.parser.add_argument('--output_size', default=1024, type=int,
                                 help='Output size of generator')

        # batch size and dataloader works
        self.parser.add_argument('--batch_size', default=4, type=int,
                                 help='Batch size for training')
        self.parser.add_argument('--test_batch_size', default=2, type=int,
                                 help='Batch size for testing and inference')
        self.parser.add_argument('--workers', default=4, type=int,
                                 help='Number of train dataloader workers')
        self.parser.add_argument('--test_workers', default=2, type=int,
                                 help='Number of test/inference dataloader workers')

        # optimizers
        self.parser.add_argument('--learning_rate', default=0.0001, type=float,
                                 help='Optimizer learning rate')
        self.parser.add_argument('--optim_name', default='ranger', type=str,
                                 help='Which optimizer to use')
        self.parser.add_argument('--train_decoder', default=False, type=bool,
                                 help='Whether to train the decoder model')

        # loss lambdas
        self.parser.add_argument('--lpips_lambda', default=0, type=float,
                                 help='LPIPS loss multiplier factor')
        self.parser.add_argument('--id_lambda', default=0, type=float,
                                 help='ID loss multiplier factor')
        self.parser.add_argument('--l2_lambda', default=0, type=float,
                                 help='L2 loss multiplier factor')
        self.parser.add_argument('--moco_lambda', default=0, type=float,
                                 help='Moco feature loss multiplier factor')

        # weights and checkpoint paths
        self.parser.add_argument('--stylegan_weights', default=model_paths["stylegan_ffhq"], type=str,
                                 help='Path to StyleGAN model weights')
        self.parser.add_argument('--checkpoint_path', default=None, type=str,
                                 help='Path to HyperStyle model checkpoint')

        # intervals for logging, validation, and saving
        self.parser.add_argument('--max_steps', default=500000, type=int,
                                 help='Maximum number of training steps')
        self.parser.add_argument('--max_val_batches', type=int, default=None,
                                 help='Number of batches to run validation on. If None, run on all batches.')
        self.parser.add_argument('--image_interval', default=100, type=int,
                                 help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=50, type=int,
                                 help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=1000, type=int,
                                 help='Validation interval')
        self.parser.add_argument('--save_interval', default=None, type=int,
                                 help='Model checkpoint interval')

        # arguments for iterative encoding
        self.parser.add_argument('--n_iters_per_batch', default=1, type=int,
                                 help='Number of forward passes per batch during training')

        # hypernet parameters
        self.parser.add_argument('--load_w_encoder', action='store_true', help='Whether to load the w e4e encoder.')
        self.parser.add_argument('--w_encoder_checkpoint_path', default=model_paths["e4e_w_encoder"], type=str,
                                 help='Path to pre-trained W-encoder.')
        self.parser.add_argument('--w_encoder_type', default='WEncoder',
                                 help='Encoder type for the encoder used to get the initial inversion')
        self.parser.add_argument('--layers_to_tune', default='0,2,3,5,6,8,9,11,12,14,15,17,18,20,21,23,24', type=str,
                                 help='comma-separated list of which layers of the StyleGAN generator to tune')

    def parse(self):
        opts = self.parser.parse_args()
        return opts
