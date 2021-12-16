""" HyperStyle Domain Adaptation cog demo

This is adapted from notebooks/domain_adaptation_playground.ipynb.

ATTN: You can run `python cog_prepare.py` to download/pre-cache model data.
You can do that via `cog run python ./notebooks/notebook_utils.py` or just run it
on your host, either way should work.
"""
import tempfile
import time
import sys
import pprint
import os
from pathlib import Path

import cog
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

sys.path.append(".")

from cog_prepare import (
    HYPERSTYLE_PATHS,
    W_ENCODERS_PATHS,
    FINETUNED_MODELS,
    RESTYLE_E4E_MODELS,
    run_alignment_offline,
    StaticDownloader,
)
from utils.common import tensor2im
from utils.domain_adaptation_utils import run_domain_adaptation
from utils.model_utils import load_model, load_generator


class Predictor(cog.Predictor):
    """HyperStyle Domain Adaption cog demo

    Reference: https://github.com/replicate/cog/blob/main/docs/python.md
    """

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        ts_start = time.time()
        self.preparer = StaticDownloader()
        self.preparer.set_torch_home()

        self.has_gpu = torch.cuda.is_available()
        if not self.has_gpu:
            raise RuntimeError("gpu is required for this predictor")

        self.models_dir = self.preparer.get_save_dir_abspath()

        model_path = self.preparer.get_hyperstyle_path()
        w_encoder_path = self.preparer.get_w_encoder_path()
        net, opts = load_model(
            model_path, update_opts={"w_encoder_checkpoint_path": w_encoder_path}
        )
        self.net = net
        self.opts = opts

        # Like in the notebooks, the default n_iters_per_batch is pegged to 5.
        self.opts.n_iters_per_batch = 5

        # At time of writing in all 3 of the notebooks, opts.resize_outputs = False.
        self.opts.resize_outputs = False

        print("Model successfully loaded")
        pprint.pprint(vars(self.opts))

        restyle_e4e_path = self.preparer.get_restyle_e4e_path()
        restyle_e4e, restyle_e4e_opts = load_model(
            restyle_e4e_path, is_restyle_encoder=True
        )
        self.restyle_e4e = restyle_e4e
        self.restyle_e4e_opts = restyle_e4e_opts
        restyle_e4e_opts.n_iters_per_batch = self.opts.n_iters_per_batch
        restyle_e4e_opts.resize_outputs = self.opts.resize_outputs
        print("ReStyle-e4e model successfully loaded!")

        ts_end = time.time()
        print("DONE. setup() took", ts_end - ts_start, "seconds")

    @cog.input("input", type=Path, help="Input image")
    @cog.input(
        "style",
        type=str,
        help="Generator type",
        options=["toonify", "pixar", "sketch", "disney_princess"],
    )
    def predict(self, image, style):
        """Run a single prediction on the model"""
        self.preparer.set_torch_home()
        print("Begin prediction")

        generator_path = self.preparer.get_generator_path(generator_type=style)
        fine_tuned_generator = load_generator(generator_path)
        print(f"Fine-tuned {style} generator successfully loaded!")

        print("Loading and preparing image")
        resize_tup = (256, 256)

        image_path = input
        img = run_alignment_offline(image_path)
        img.resize(resize_tup)

        transform = transforms.Compose(
            [
                transforms.Resize(resize_tup),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        transformed_image = transform(img)

        with torch.no_grad():
            tic = time.time()
            result, _ = run_domain_adaptation(
                transformed_image.unsqueeze(0).cuda(),
                self.net,
                self.opts,
                fine_tuned_generator,
                self.restyle_e4e,
                self.restyle_e4e_opts,
            )
            toc = time.time()
            print("Inference took {:.4f} seconds.".format(toc - tic))

        resize_amount = (
            resize_tup
            if self.opts.resize_outputs
            else (self.opts.output_size, self.opts.output_size)
        )

        final_res = tensor2im(result[0]).resize(resize_amount)
        input_im = tensor2im(transformed_image).resize(resize_amount)
        res = np.concatenate([np.array(input_im), np.array(final_res)], axis=1)
        res = Image.fromarray(res, mode="RGB")

        outputs_dir = Path(tempfile.mkdtemp())
        os.makedirs(outputs_dir, exist_ok=True)
        output_path = outputs_dir / "out.png"
        res.save(str(output_path))

        assert isinstance(output_path, Path), "cog expects a Path (not a string)"
        return output_path
