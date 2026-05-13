import os

import bardensr
import numpy as np
import pandas as pd
import skimage.io
import tensorflow as tf


def main(args):
    cb_df = pd.read_csv(args.codebook_file, index_col="target", sep="\t")

    cb = cb_df.values.T

    imgs = skimage.io.imread(args.imgs_file).astype(float)

    imgs = imgs[:, np.newaxis, ...]

    imgs_norm = bardensr.preprocessing.minmax(imgs)

    imgs_norm = bardensr.preprocessing.background_subtraction(imgs_norm, [0, 10, 10])

    imgs_norm = bardensr.preprocessing.minmax(imgs_norm)

    evidence_tensor_iterative, extra_learned_params = bardensr.spot_calling.estimate_density_iterative(
        imgs_norm, cb.astype(int), iterations=100, use_tqdm_notebook=False
    )

    thresh_iterative = evidence_tensor_iterative.max() * 0.1

    out_df = bardensr.spot_calling.find_peaks(evidence_tensor_iterative, thresh_iterative)

    out_df.columns = "z", "x", "y", "target_id"

    out_df["evidence"] = out_df.apply(lambda row: evidence_tensor_iterative[tuple(row.values)], axis=1)

    out_df["target"] = out_df.apply(lambda row: cb_df.index[row["target_id"].astype(int)], axis=1)

    out_df.to_csv(args.out_file, index=False, sep="\t")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--codebook-file", required=True)

    parser.add_argument("-i", "--imgs-file", required=True)

    parser.add_argument("-o", "--out-file", required=True)

    parser.add_argument("-t", "--num-threads", default=1, type=int)

    cli_args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = f"{cli_args.num_threads}"
    tf.config.threading.set_intra_op_parallelism_threads(cli_args.num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(cli_args.num_threads)

    main(cli_args)
