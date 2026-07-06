import os
import pathlib
import tempfile

import numba
import pandas as pd

from savannah.run import run_fit, run_predict, run_preprocess


def main(args):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = pathlib.Path(tmp_dir)

        pp_file = tmp_dir.joinpath("pp.tif")

        fit_file = tmp_dir.joinpath("fit.h5")

        pred_file = tmp_dir.joinpath("spots.tsv.gz")

        run_preprocess(
            in_file=args.imgs_file,
            out_file=str(pp_file),
            clip=args.clip,
            decon_sigma=args.decon_sigma,
            high_pass_sigma=args.high_pass_sigma,
            decon_filter_size=args.decon_filter_size,
            low_pass_sigma=args.low_pass_sigma,
            num_decon_iters=args.num_decon_iters,
            subtract_background=args.subtract_background,
        )

        run_fit(
            codebook_file=args.codebook_file,
            img_path=str(pp_file),
            out_file=str(fit_file),
            background_prior=args.background_prior,
            batch_size=args.batch_size,
            crop_size=args.crop_size,
            convergence_threshold=args.convergence_threshold,
            expression_prior_file=args.expression_prior_file,
            num_restarts=args.num_restarts,
            max_batch_iters=args.max_batch_iters,
            max_svi_iters=args.max_svi_iters,
            outlier_prior=args.outlier_prior,
            print_freq=args.print_freq,
            seed=args.seed,
            transform=args.transform,
            use_gmm=args.use_gmm,
        )

        run_predict(
            fit_files=[str(fit_file)],
            img_file=str(pp_file),
            out_file=str(pred_file),
            compute_elbo=args.compute_elbo,
            crop_size=args.crop_size,
            max_area=args.max_area,
            max_dist=args.max_dist,
            min_area=args.min_area,
            min_prob=args.min_prob,
            pred_file=None,
            table_format="spot",
        )

        out_df = pd.read_csv(pred_file, sep="\t")

    # Savannah reports spot centre as centroid-0 (y) / centroid-1 (x)
    out_df = out_df.rename(columns={"centroid-0": "y", "centroid-1": "x"})

    out_df.to_csv(args.out_file, index=False, sep="\t")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--codebook-file", required=True)

    parser.add_argument("-i", "--imgs-file", required=True)

    parser.add_argument("-o", "--out-file", required=True)

    parser.add_argument("-t", "--num-threads", default=1, type=int)

    # preprocess params
    parser.add_argument("--clip", action="store_true")

    parser.add_argument("--decon-sigma", default=2, type=float)

    parser.add_argument("--decon-filter-size", default=None, type=int)

    parser.add_argument("--high-pass-sigma", default=3, type=float)

    parser.add_argument("--low-pass-sigma", default=1, type=float)

    parser.add_argument("--num-decon-iters", default=15, type=int)

    parser.add_argument("--subtract-background", action="store_true")

    # fit params
    parser.add_argument("--background-prior", default=1000, type=float)

    parser.add_argument("--batch-size", default=1000, type=int)

    parser.add_argument("--crop-size", default=0, type=int)

    parser.add_argument("--convergence-threshold", default=1e-6, type=float)

    parser.add_argument("--expression-prior-file", default=None)

    parser.add_argument("--num-restarts", default=10, type=int)

    parser.add_argument("--max-batch-iters", default=1, type=int)

    parser.add_argument("--max-svi-iters", default=int(1e4), type=int)

    parser.add_argument("--outlier-prior", default=0, type=float)

    parser.add_argument("--print-freq", default=10, type=int)

    parser.add_argument("--seed", default=None, type=int)

    parser.add_argument("--transform", default="none", choices=["none", "boxcox", "log", "std"])

    parser.add_argument("--use-gmm", action="store_true")

    # predict params
    parser.add_argument("--compute-elbo", action="store_true")

    parser.add_argument("--max-area", default=1000, type=int)

    parser.add_argument("--max-dist", default=None, type=int)

    parser.add_argument("--min-area", default=1, type=int)

    parser.add_argument("--min-prob", default=0.99, type=float)

    cli_args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = f"{cli_args.num_threads}"
    numba.set_num_threads(cli_args.num_threads)

    main(cli_args)