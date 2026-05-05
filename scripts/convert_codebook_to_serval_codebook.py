import pandas as pd


def main(args):
    cb_df = pd.read_csv(args.codebook_file, index_col="target", sep="\t")

    if "dist" in cb_df.columns:
        cb_df = cb_df.drop(columns="dist")

    do_df = pd.read_csv(args.data_org_file, converters={"bit_number": int}, sep="\t")

    do_df = do_df.sort_values(by="bit_number")

    cb_df = cb_df[do_df["bit_id"]]

    cb_df.to_csv(args.out_file, sep="\t")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--codebook-file", required=True)

    parser.add_argument("-d", "--data-org-file", required=True)

    parser.add_argument("-o", "--out-file", required=True)

    cli_args = parser.parse_args()

    main(cli_args)
