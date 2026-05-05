import matplotlib.pyplot as pp
import numpy as np
import pandas as pd


def main(args):
    in_files = dict(zip(args.decoders, args.in_files))

    fig = pp.figure(figsize=(12, 4))

    ax_r = fig.add_subplot(1, 2, 1)

    ax_rho = fig.add_subplot(1, 2, 2)

    for decoder, file_name in in_files.items():
        df = pd.read_csv(file_name, sep="\t")

        ax_r.plot(df["num_emitters"], df["r"], label=decoder)

        ax_rho.plot(df["num_emitters"], df["rho"], label=decoder)

    ax_r.set_xlabel("Number of emitters")
    ax_r.set_ylabel("Pearson r")
    ax_r.legend()
    ax_r.set_ylim(0, 1)

    ax_rho.set_xlabel("Number of emitters")
    ax_rho.set_ylabel("Spearman rho")
    ax_rho.legend()
    ax_rho.set_ylim(0, 1)

    fig.savefig(args.out_file, bbox_inches="tight")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--decoders", nargs="+", required=True)

    parser.add_argument("-i", "--in-files", nargs="+", required=True)

    parser.add_argument("-o", "--out-file", required=True)

    cli_args = parser.parse_args()

    main(cli_args)
