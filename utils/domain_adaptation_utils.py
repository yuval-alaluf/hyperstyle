import sys
sys.path.extend(['.', '..'])

from utils.inference_utils import run_inversion
from utils import restyle_inference_utils


def run_domain_adaptation(inputs, net, opts, fine_tuned_generator, restyle_e4e, restyle_opts):
    """ Combine restyle e4e's latent code with HyperStyle's predicted weight offsets. """
    y_hat, latents = restyle_inference_utils.run_on_batch(inputs, restyle_e4e, restyle_opts)
    y_hat2, _, weights_deltas, _ = run_inversion(inputs, net, opts)
    weights_deltas = filter_non_ffhq_layers_in_toonify_model(weights_deltas)
    return fine_tuned_generator([latents],
                                input_is_latent=True,
                                randomize_noise=True,
                                return_latents=True,
                                weights_deltas=weights_deltas)


def filter_non_ffhq_layers_in_toonify_model(weights_deltas):
    toonify_ffhq_layer_idx = [14, 15, 17, 18, 20, 21, 23, 24]  # convs 8-15 according to model_utils.py
    for i in range(len(weights_deltas)):
        if weights_deltas[i] is not None and i not in toonify_ffhq_layer_idx:
            weights_deltas[i] = None
    return weights_deltas

