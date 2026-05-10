import matplotlib.pyplot as pp
import pandas as pd


def main(args):
    in_files = dict(zip(args.decoders, args.in_files))

    fig = pp.figure(figsize=(12, 4))

    ax_loc = fig.add_subplot(1, 2, 1)

    ax_exc = fig.add_subplot(1, 2, 2)

    for decoder, file_name in in_files.items():
        df = pd.read_csv(file_name, sep="\t")

        ax_loc.plot(df["loc_recall"], df["loc_precision"], label=decoder)

        ax_exc.plot(df["exc_recall"], df["exc_precision"], label=decoder)

    ax_loc.set_title("Location")
    ax_loc.set_xlabel("Recall")
    ax_loc.set_ylabel("Precision")
    ax_loc.legend()
    ax_loc.set_ylim(0, 1)

    ax_exc.set_title("Exact")
    ax_loc.set_xlabel("Recall")
    ax_loc.set_ylabel("Precision")
    ax_exc.legend()
    ax_exc.set_ylim(0, 1)

    fig.savefig(args.out_file, bbox_inches="tight")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--decoders", nargs="+", required=True)

    parser.add_argument("-i", "--in-files", nargs="+", required=True)

    parser.add_argument("-o", "--out-file", required=True)

    cli_args = parser.parse_args()

    main(cli_args)
