import pandas as pd


def main(args):
    header = True

    mode = "w"

    for file_name in args.in_files:
        df = pd.read_csv(file_name, sep="\t")

        df.to_csv(args.out_file, header=header, index=False, mode=mode, sep="\t")

        header = False

        mode = "a"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--in-files", nargs="+", required=True)

    parser.add_argument("-o", "--out-file", required=True)

    cli_args = parser.parse_args()

    main(cli_args)
