import numpy as np
import oct2py


def main(args):
    scaled_patch_size = args.patch_size * args.scale_factor

    oc = oct2py.Oct2Py()

    oc.addpath(args.jsit_src_path)

    oc.eval("pkg load image")

    psf = oc.getPsfMat2(scaled_patch_size, args.scale_factor, args.sigma)

    np.save(args.out_file, psf)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-j", "--jsit-src-path", required=True)

    parser.add_argument("-p", "--patch-size", default=32, type=int)

    parser.add_argument("-s", "--scale-factor", default=3, type=int)

    parser.add_argument("-o", "--out-file", required=True)

    parser.add_argument("--sigma", default=1.25, type=float)

    cli_args = parser.parse_args()

    main(cli_args)
