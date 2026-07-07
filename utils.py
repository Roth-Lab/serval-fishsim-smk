import pathlib


class ConfigManager(object):
    def __init__(self, config):
        self.config = config

    @property
    def pipeline_files(self):
        files = [self.bulk_metrics_file, self.emitter_metrics_file, self.summary_metrics_file]
        # for run in self.runs:
        #     for rep in range(self.num_replicates):
        #         for score in self.scores:
        #             files.append(str(self.bulk_metrics_plot_template).format(replicate=rep, run=run, score=score))
        #             files.append(str(self.emitter_metrics_plot_template).format(replicate=rep, run=run, score=score))
        return files

    # Input files
    @property
    def codebook_file(self):
        return pathlib.Path(self.config["codebook_file"])

    @property
    def data_org_file(self):
        return pathlib.Path(self.config["data_org_file"])

    @property
    def deepcell_model_path(self):
        return pathlib.Path("resources/deepcell/SpotDetection-8")

    @property
    def jsit_src_dir(self):
        return pathlib.Path("resources/jsit/src")

    @property
    def run_config_dir(self):
        return pathlib.Path(self.config["run_config_dir"])

    # Params
    @property
    def deepcell_threads(self):
        return self.config["deepcell"]["num_threads"]
    
    # Params
    @property
    def savannah_threads(self):
        return self.config["savannah"]["num_threads"]

    # Params
    @property
    def bardensr_threads(self):
        return self.config["bardensr"]["num_threads"]

    @property
    def decoders(self):
        # return ["bardensr", "deepcell", "jsit", "cosine", "cosine-np", "nn", "scaled"]
        return ["bardensr", "deepcell", "cosine", "cosine-np", "nn", "scaled", "savannah"]

    @property
    def jsit_patch_size(self):
        return self.config["jsit"]["patch_size"]

    @property
    def jsit_penalties(self):
        return [0.01, 1, 50, 100, 150, 500]

    @property
    def jsit_thresholds(self):
        return [0, 0.1, 0.2]

    @property
    def jsit_scale_factor(self):
        return self.config["jsit"]["scale_factor"]

    @property
    def jsit_threads(self):
        return self.config["jsit"]["num_threads"]

    @property
    def num_replicates(self):
        return self.config.get("num_replicates", 1)

    @property
    def runs(self):
        return self.config["runs"]

    # Directories
    @property
    def out_dir(self):
        return pathlib.Path(self.config["out_dir"])

    @property
    def tmp_dir(self):
        return self.out_dir.joinpath("tmp")

    # Pipeline files
    @property
    def bulk_metrics_file(self):
        return self.out_dir.joinpath("bulk_metrics.tsv.gz")

    @property
    def bulk_metrics_template(self):
        return self.tmp_dir.joinpath("{run}", "{replicate}", "bulk_metrics", "{decoder}.tsv.gz")

    @property
    def bulk_metrics_plot_template(self):
        return self.out_dir.joinpath("plots", "bulk", "{run}", "{replicate}.png")

    @property
    def emitter_metrics_file(self):
        return self.out_dir.joinpath("emitter_metrics.tsv.gz")

    @property
    def emitter_metrics_template(self):
        return self.tmp_dir.joinpath("{run}", "{replicate}", "emitter_metrics", "{decoder}.tsv.gz")

    @property
    def emitter_metrics_plot_template(self):
        return self.out_dir.joinpath("plots", "emitter", "{run}", "{replicate}.png")

    @property
    def jsit_emitter_metrics_template(self):
        return self.tmp_dir.joinpath(
            "{run}", "{replicate}", "jsit", "{jsit_penalty}", "{jsit_threshold}", "emitter_metrics.tsv.gz"
        )

    @property
    def jsit_spots_template(self):
        return self.tmp_dir.joinpath(
            "{run}", "{replicate}", "jsit", "{jsit_penalty}", "{jsit_threshold}", "spots.tsv.gz"
        )

    @property
    def jsit_psf_file(self):
        return self.tmp_dir.joinpath("jsit_psf.npy")

    @property
    def run_config_template(self):
        return self.run_config_dir.joinpath("{run}.yaml")

    @property
    def serval_codebook_file(self):
        return self.out_dir.joinpath("serval_codebook.tsv")

    @property
    def sim_emitter_template(self):
        return self.tmp_dir.joinpath("{run}", "{replicate}", "emitters.tsv.gz")

    @property
    def sim_img_template(self):
        return self.tmp_dir.joinpath("{run}", "{replicate}", "spot_img.tif")

    @property
    def sim_pp_img_template(self):
        return self.tmp_dir.joinpath("{run}", "{replicate}", "pp_spot_img.tif")

    @property
    def spots_template(self):
        return self.tmp_dir.joinpath("{run}", "{replicate}", "spots", "{decoder}.tsv.gz")

    @property
    def summary_metrics_file(self):
        return self.out_dir.joinpath("summary_metrics.tsv.gz")

    @property
    def summary_metrics_template(self):
        return self.tmp_dir.joinpath("{run}", "{replicate}", "summary_metrics", "{decoder}.tsv.gz")

    # Helper function
    def get_decoder_args(self, wildcards):
        args = []

        if wildcards.decoder == "cosine-np":
            args.extend(["--penalty-entropy 0", "--penalty-l2 0"])

            decoder = "cosine"

        else:
            decoder = wildcards.decoder

        args.append(f"--decoder {decoder}")

        return " ".join(args)
