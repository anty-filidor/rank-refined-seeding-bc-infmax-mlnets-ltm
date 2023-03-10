import argparse
import yaml

from misc.utils import set_seed
from runners import runner_default, runner_greedy, runner_optimised


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Experiment config file (default: config.yaml).",
        type=str,
        default="data/example_configs/example_nghb_s.yaml",
    )
    parser.add_argument(
        "--runner",
        help="A runner function to execute (default: default_runner).",
        type=str,
        default="runner_default",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if _ := config["run"].get("random_seed"):
        print(f"Setting random seed as {_}!")
        set_seed(config["run"]["random_seed"])
    print(f"Loaded config: {config}")

    if args.runner == "runner_default":
        print(f"Inferred runner as: {args.runner}")
        runner_default.run_experiments(config)
    elif args.runner == "runner_greedy":
        print(f"Inferred runner as: {args.runner}")
        runner_greedy.run_experiments(config)
    elif args.runner == "runner_optimised":
        print(f"Inferred runner as: {args.runner}")
        runner_optimised.run_experiments(config)
    else:
        raise AttributeError(f"Incorrect runner name ({args.runner})!")