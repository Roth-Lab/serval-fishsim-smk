from dask.distributed import Client

import dask
import pandas as pd
import skimage

from serval.codebook import Codebook
from serval.decode.pixel import CosineOptimizedPixelDecoder, NearestNeigbourPixelDecoder, ScaledImagePixelDecoder
from serval.decode.utils import get_imgs_hist, get_init_scaling_factors
from serval.image import ImageStack
from serval.pipeline import DaskDecodingPipeline

import serval.transform


def main(args):
    assert args.min_area <= args.max_area

    codebook = load_codebook(args.codebook_file)

    transforms = [
        serval.transform.HighPassImageTransform(sigma=args.high_pass_sigma),
        serval.transform.DeconvoleImageTransform(
            filter_size=args.deconv_filter_size, num_iters=args.deconv_iters, sigma=args.deconv_sigma
        ),
        serval.transform.LowPassImageTransform(sigma=args.low_pass_sigma),
    ]

    imgs = load_spot_imgs(args.imgs_file)

    init_scaling_factors = load_init_scaling_factors(imgs, transforms)

    print(init_scaling_factors)

    nn = NearestNeigbourPixelDecoder(
        codebook,
        max_dist=0.5167,
        min_norm=1,
    )

    if args.decoder == "cosine":
        decoder = CosineOptimizedPixelDecoder(
            codebook,
            nn,
            max_area=args.max_area,
            min_area=args.min_area,
            penalty_entropy=args.penalty_entropy,
            penalty_l2=args.penalty_l2,
            pre_scaling_factors=init_scaling_factors,
        )

    elif args.decoder == "scaled":
        decoder = ScaledImagePixelDecoder(
            codebook,
            nn,
            init_scaling_factors=init_scaling_factors,
            max_area=args.max_area,
            min_area=args.min_area,
        )

    else:
        decoder = nn

    pipeline = DaskDecodingPipeline(
        decoder, transforms, fit_num_iters=args.num_iters
    )

    pipeline.fit([imgs])

    df = pipeline.predict([imgs])[0].spots

    df = dask.compute(df)[0]

    df.to_csv(args.out_file, index=False, sep="\t")


def load_codebook(file_name):
    df = pd.read_csv(file_name, index_col="target", sep="\t")

    return Codebook(df)


def load_spot_imgs(imgs_file):
    return ImageStack(
        skimage.io.imread(imgs_file),
        fov=0,
        z=0
    )


def load_init_scaling_factors(imgs, transforms):
    hists = get_imgs_hist(imgs, transforms)

    return get_init_scaling_factors(hists)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--codebook-file", required=True)

    parser.add_argument("-i", "--imgs-file", required=True)

    parser.add_argument("-o", "--out-file", required=True)

    parser.add_argument("--cores", default=1, type=int)

    parser.add_argument("--decoder", choices=["cosine", "nn", "scaled"], default="cosine")

    parser.add_argument("--deconv-iters", default=10, type=int)

    parser.add_argument("--deconv-filter-size", default=9, type=int)

    parser.add_argument("--deconv-sigma", default=1.2, type=float)

    parser.add_argument("--high-pass-sigma", default=2, type=float)

    parser.add_argument("--low-pass-sigma", default=1, type=float)

    parser.add_argument("--max-area", default=12, type=int)

    parser.add_argument("--min-area", default=2, type=int)

    parser.add_argument("--penalty-entropy", default=1e-2, type=float)

    parser.add_argument("--penalty-l2", default=1e-3, type=float)

    parser.add_argument("--num-iters", default=10, type=int)

    cli_args = parser.parse_args()

    main(cli_args)
