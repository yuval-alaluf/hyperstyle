"""
This file runs the main training/val loop
"""
import json
import os
import sys
import pprint

sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from training.coach_hyperstyle import Coach


def main():
	opts = TrainOptions().parse()
	create_initial_experiment_dir(opts)
	coach = Coach(opts)
	coach.train()


def create_initial_experiment_dir(opts):
	os.makedirs(opts.exp_dir, exist_ok=True)
	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)


if __name__ == '__main__':
	main()
