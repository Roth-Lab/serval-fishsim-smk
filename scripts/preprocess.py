from serval.image import ImageStack
from serval.pipeline import SerialDecodingPipeline

import serval.transform
import skimage


def main(args):
    imgs = ImageStack.load(args.in_file)

    transforms = [
        serval.transform.HighPassImageTransform(sigma=args.high_pass_sigma),
        serval.transform.DeconvoleImageTransform(
            filter_size=args.deconv_filter_size, num_iters=args.deconv_iters, sigma=args.deconv_sigma
        ),
        serval.transform.LowPassImageTransform(sigma=args.low_pass_sigma),
    ]

    pipeline = SerialDecodingPipeline(None, transforms, transform_preserve_dtype=True)

    imgs = pipeline.transform([imgs])[0]

    skimage.io.imsave(args.out_file, imgs.imgs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--in-file", required=True)

    parser.add_argument("-o", "--out-file", required=True)

    parser.add_argument("--deconv-iters", default=10, type=int)

    parser.add_argument("--deconv-filter-size", default=9, type=int)

    parser.add_argument("--deconv-sigma", default=1.2, type=float)

    parser.add_argument("--high-pass-sigma", default=2, type=float)

    parser.add_argument("--low-pass-sigma", default=1, type=float)

    cli_args = parser.parse_args()

    main(cli_args)
