import os

from deepcell_spots.applications import SpotDecoding, SpotDetection
from deepcell_spots.dotnet_losses import DotNetLosses
from deepcell_spots.multiplex import extract_spots_prob_from_coords_maxpool
from deepcell_spots.utils.postprocessing_utils import max_cp_array_to_point_list_max
from tensorflow.keras.utils import get_file

import numpy as np
import pandas as pd
import skimage
import tensorflow as tf


def main(args):
    codebook_df, imgs, num_channels, num_rounds = load_data(args.codebook_file, args.data_org_file, args.imgs_file)

    model = load_model(args.model_path)

    spot_int_vec, spot_loc_vec = call_spots(imgs, model)

    decode_result = decode_spots(codebook_df, num_channels, num_rounds, spot_int_vec)

    df = pd.DataFrame()

    df[["x", "y", "batch_id"]] = spot_loc_vec.astype(np.int32)

    for name, val in decode_result.items():
        df[name] = val

    df = df.rename(columns={"predicted_name": "target"})

    df.to_csv(args.out_file, index=False, sep="\t")


def call_spots(imgs, model, spots_threshold=0.9):
    app = SpotDetection(model)

    app.postprocessing_fn = None

    pred = app.predict(imgs, clip=True, threshold=0.5)

    output_imgs = pred["classification"][:, ..., 1:2]

    output_imgs = np.swapaxes(output_imgs, 0, 3)

    max_proj_imgs = np.max(output_imgs, axis=-1)

    spots_locs = max_cp_array_to_point_list_max(max_proj_imgs, threshold=spots_threshold, min_distance=1)

    maxpool_extra_pixel_num = 0

    spot_int = extract_spots_prob_from_coords_maxpool(
        output_imgs,
        spots_locs,
        extra_pixel_num=maxpool_extra_pixel_num,
    )

    spot_int_vec = np.concatenate(spot_int)

    spot_loc_vec = np.concatenate(
        [np.concatenate([item, [[idx_batch]] * len(item)], axis=1) for idx_batch, item in enumerate(spots_locs)]
    )

    return spot_int_vec, spot_loc_vec


def decode_spots(codebook_df, num_channels, num_rounds, spot_int_vec):
    app = SpotDecoding(codebook_df, channels=num_channels, rounds=num_rounds)

    return app.predict(np.clip(spot_int_vec, a_min=0, a_max=1))


def load_data(codebook_file, data_org_file, imgs_file):
    data_org_df = pd.read_csv(data_org_file, sep="\t")

    num_channels = data_org_df["channel"].nunique()

    num_rounds = data_org_df["imaging_round"].nunique()

    bits = list(data_org_df.sort_values(by=["imaging_round", "channel"])["bit_id"].values)

    codebook_df = pd.read_csv(codebook_file, index_col="target", sep="\t")

    imgs_idxs = [bits.index(x) for x in codebook_df.columns]

    codebook_df = codebook_df[bits].reset_index().rename(columns={"target": "Gene"})

    imgs = skimage.io.imread(imgs_file)

    # Reorder image stack by round/channel
    imgs = imgs[imgs_idxs]

    # Pad last axis
    imgs = imgs[..., np.newaxis]

    return codebook_df, imgs, num_channels, num_rounds


def load_model(path):
    return tf.keras.models.load_model(
        path,
        custom_objects={
            "regression_loss": DotNetLosses.regression_loss,
            "classification_loss": DotNetLosses.classification_loss,
        },
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--codebook-file", required=True)

    parser.add_argument("-d", "--data-org-file", required=True)

    parser.add_argument("-i", "--imgs-file", required=True)

    parser.add_argument("-m", "--model-path", required=True)

    parser.add_argument("-o", "--out-file", required=True)

    parser.add_argument("-t", "--num-threads", default=1, type=int)

    cli_args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = f"{cli_args.num_threads}"
    tf.config.threading.set_intra_op_parallelism_threads(cli_args.num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(cli_args.num_threads)

    main(cli_args)
