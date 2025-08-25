import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from ..train_util import create_losses_plot

OBJS_DIR = "./outs/binary_classification/spambase/objs/"
FIG_DIR = "./outs/binary_classification/spambase/figures/"


def main():
    parser = argparse.ArgumentParser(prog="plots")
    parser.add_argument("--save-fig", action="store_true")
    parser.add_argument(
        "-s",
        "--step_size",
        choices=["decaying", "strong_wolfe", "prob_wolfe"],
        default="decaying",
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")

    sgd_out = torch.load(OBJS_DIR + "sgd-1.pt")
    olbfgs_out = torch.load(OBJS_DIR + f"olbfgs-{args.step_size}-1.pt")
    sqnhv_out = torch.load(OBJS_DIR + f"sqnhv-{args.step_size}-1.pt")
    mbbfgs_out = torch.load(OBJS_DIR + f"mbbfgs-{args.step_size}-1.pt")
    scbfgs_out = torch.load(OBJS_DIR + f"scbfgs-{args.step_size}-1.pt")

    fig, ax = plt.subplots(figsize=(8, 6))
    create_losses_plot(
        ax,
        (sgd_out, olbfgs_out, sqnhv_out, mbbfgs_out, scbfgs_out),
        ("SGD", "oL-BFGS", "SQN-Hv", "MB-BFGS", "SC-BFGS"),
    )

    if args.save_fig:
        plt.savefig(FIG_DIR + "spambase-loss.pdf")
    else:
        plt.show()


if __name__ == "__main__":
    main()
