import pathlib


class ConfigManager(object):
    def __init__(self, config):
        self.config = config

    @property
    def pipeline_files(self):
        files = [self.bulk_metrics_file, self.emitter_metrics_file, self.summary_metrics_file]
        for run in self.runs:
            for rep in range(self.num_replicates):
                files.append(str(self.bulk_metrics_plot_template).format(replicate=rep, run=run))
                files.append(str(self.emitter_metrics_plot_template).format(replicate=rep, run=run))
        return files

    # Input files
    @property
    def codebook_file(self):
        return pathlib.Path(self.config["codebook_file"])

    @property
    def data_org_file(self):
        return pathlib.Path(self.config["data_org_file"])

    @property
    def run_config_dir(self):
        return pathlib.Path(self.config["run_config_dir"])

    # Params
    @property
    def decoders(self):
        return ["cosine", "nn", "scaled"]

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
    def spots_template(self):
        return self.tmp_dir.joinpath("{run}", "{replicate}", "spots", "{decoder}.tsv.gz")

    @property
    def summary_metrics_file(self):
        return self.out_dir.joinpath("summary_metrics.tsv.gz")

    @property
    def summary_metrics_template(self):
        return self.tmp_dir.joinpath("{run}", "{replicate}", "summary_metrics", "{decoder}.tsv.gz")
