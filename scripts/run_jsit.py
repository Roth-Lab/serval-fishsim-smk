import logging

from threadpoolctl import threadpool_limits

import numba
import numpy as np
import oct2py
import pandas as pd
import skimage


def main(args):
    numba.set_num_threads(args.num_threads)

    scaled_patch_size = args.patch_size * args.scale_factor

    logging.basicConfig(level=logging.INFO)

    oc = oct2py.Oct2Py(logger=logging.getLogger())

    oc.addpath(args.jsit_src_path)

    oc.eval("pkg load image")

    cb_df = pd.read_csv(args.codebook_file, index_col="target", sep="\t")

    cb = cb_df.values.astype(float)

    imgs = skimage.io.imread(args.imgs_file).astype(float)

    imgs = np.moveaxis(imgs, 0, -1)

    A = np.load(args.psf_file)

    lf = oc.svd(cb @ cb.T)[0, 0] * oc.svd(A @ A.T)[0, 0]

    y_stack = oc.makeYstack(imgs, args.patch_size, args.patch_size)

    num_tiles = y_stack.shape[2]

    x_stack = np.zeros((num_tiles, scaled_patch_size**2, cb.shape[0]))

    with threadpool_limits(limits=args.num_threads, user_api="blas"):
        for i in range(num_tiles):
            x_stack[i] = codebook_fista(
                A, cb, y_stack[:, :, i], lf, alpha=args.alpha, max_iters=args.max_iters, penalty=args.penalty
            )

    x_stack = process_x_stack(
        x_stack,
        imgs.shape[0] // args.patch_size,
        imgs.shape[1] // args.patch_size,
        scaled_patch_size,
        scaled_patch_size,
    )

    x_stack = enforce_sparsity(x_stack, k=1, t=args.sparsity_threshold)

    scaled_img_x = imgs.shape[0] * args.scale_factor

    scaled_img_y = imgs.shape[1] * args.scale_factor

    d_img = x_to_dimg(x_stack, scaled_img_x, scaled_img_y)

    x_img = x_stack.sum(axis=1).reshape((scaled_img_x, scaled_img_y))

    i_img = np.linalg.norm(imgs.reshape((imgs.shape[0] * imgs.shape[1], imgs.shape[2])), axis=1, ord=2).reshape(
        (imgs.shape[0], imgs.shape[1])
    )

    i_img = oc.imresize(i_img, args.scale_factor)

    try:
        out_df = oc.dIm2q_ex(d_img, i_img, x_img, args.min_area, cb)

        out_df[:, :2] = out_df[:, :2] / args.scale_factor

        out_df = pd.DataFrame(out_df, columns=["y", "x", "barcode_id", "area", "mean_intensity", "mean_magnitude"])

        out_df["barcode_id"] = out_df["barcode_id"].astype(int) - 1

        out_df["area"] = out_df["area"].astype(int)

        out_df["target"] = cb_df.index[out_df["barcode_id"]]

        out_df["x"] = out_df["x"] - 1

        out_df["y"] = out_df["y"] - 1

    except oct2py.utils.Oct2PyError:
        out_df = pd.DataFrame([], columns=["y", "x", "barcode_id", "area", "mean_intensity", "mean_magnitude"])

    out_df.to_csv(args.out_file, index=False, sep="\t")


@numba.njit
def codebook_fista(A, C, Y, lf, alpha=0.5, max_iters=10, penalty=1):
    s1 = A.shape[1]
    s2 = C.shape[0]
    z = np.zeros((s1, s2))
    X = np.zeros((s1, s2))
    t = 1
    for k in range(max_iters):
        gfz = A.T @ (A @ z @ C - Y) @ C.T
        Xp = X
        X = prox_sgl(alpha, penalty / lf, X - (gfz / lf))
        tp = t
        t = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        z = X + (t / tp) * (X - Xp)
    return X


@numba.njit
def prox_sgl(alpha, t, x):
    x = soft_threshold(x, alpha * t)
    y = np.zeros(x.shape)
    for r in range(x.shape[0]):
        norm = np.linalg.norm(x[r], ord=2)
        if norm >= t * (1 - alpha):
            y[r] = (1 - (t * (1 - alpha) / norm)) * x[r]
    return y


@numba.njit
def soft_threshold(x, t):
    return np.sign(x) * np.fmax(np.abs(x) - t, 0)


def enforce_sparsity(x, k=1, t=0):
    x_sparse = np.zeros(x.shape)
    for i in range(x.shape[0]):
        idxs = np.argsort(x[i])[::-1]
        for j in range(k):
            if x[i, idxs[j]] >= t:
                x_sparse[i, idxs[j]] = x[i, idxs[j]]
    return x_sparse


def process_x_stack(x, d1, d2, h1, h2):
    n, _, g = x.shape
    x = x.reshape((n, h1, h2, g))
    temp = []
    for i in range(d1):
        temp.append(np.concatenate([x[d1 * j + i] for j in range(d2)], axis=1))
    temp = np.concatenate(temp)
    return temp.reshape((d1 * d2 * h1 * h2, g))


def x_to_dimg(x, s1, s2):
    d_img = x.argmax(axis=1) + 1
    d_img[x.max(axis=1) == 0] = 0
    d_img = d_img.reshape((s1, s2))
    return d_img.T


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--codebook-file", required=True)

    parser.add_argument("-i", "--imgs-file", required=True)

    parser.add_argument("-j", "--jsit-src-path", required=True)

    parser.add_argument("-p", "--psf-file", required=True)

    parser.add_argument("-o", "--out-file", required=True)

    parser.add_argument("-t", "--num-threads", default=1, type=int)

    parser.add_argument("--alpha", default=0.5, type=float)

    parser.add_argument("--max-iters", default=10, type=int)

    parser.add_argument("--min-area", default=2, type=int)

    parser.add_argument("--patch-size", default=32, type=int)

    parser.add_argument("--penalty", default=75, type=float)

    parser.add_argument("--scale-factor", default=3, type=int)

    parser.add_argument("--sparsity-threshold", default=0, type=float)

    cli_args = parser.parse_args()

    main(cli_args)
