from utils import ConfigManager

config = ConfigManager(config)


rule all:
    input:
        config.pipeline_files,


rule simulate_data:
    input:
        b=config.codebook_file,
        c=config.run_config_template,
        d=config.data_org_file,
    output:
        e=config.sim_emitter_template,
        i=config.sim_img_template,
    conda:
        "envs/fishsim.yaml"
    shell:
        "fishsim simulate -b {input.b} -c {input.c} -d {input.d} -e {output.e} -i {output.i} --seed {wildcards.replicate}"


rule serval_codebook:
    input:
        c=config.codebook_file,
        d=config.data_org_file,
    output:
        config.serval_codebook_file,
    conda:
        "envs/python.yaml"
    shell:
        "python scripts/convert_codebook_to_serval_codebook.py -c {input.c} -d {input.d} -o {output}"


rule fit_serval:
    input:
        c=config.serval_codebook_file,
        i=config.sim_img_template,
    output:
        config.spots_template,
    conda:
        "envs/serval.yaml"
    shell:
        "python scripts/fit.py "
        "-c {input.c} "
        "-i {input.i} "
        "-o {output} "
        "--decoder {wildcards.decoder}"


rule compute_bulk_metrics:
    input:
        p=config.spots_template,
        t=config.sim_emitter_template,
    output:
        config.bulk_metrics_template,
    conda:
        "envs/python.yaml"
    shell:
        "python scripts/compute_bulk_metrics.py "
        "-p {input.p} "
        "-t {input.t} "
        "-o {output} "
        "--decoder {wildcards.decoder} "
        "--replicate {wildcards.replicate} "
        "--run {wildcards.run} "
        "--score {wildcards.score} "


rule plot_bulk_metrics:
    input:
        expand(
            config.bulk_metrics_template, decoder=config.decoders, allow_missing=True
        ),
    output:
        config.bulk_metrics_plot_template,
    params:
        " ".join(config.decoders),
    conda:
        "envs/python.yaml"
    shell:
        "python scripts/plot_bulk_metrics.py -d {params} -i {input} -o {output}"


rule merge_bulk_metrics:
    input:
        expand(
            config.bulk_metrics_template,
            decoder=config.decoders,
            replicate=range(config.num_replicates),
            run=config.runs,
            score=config.scores,
        ),
    output:
        config.bulk_metrics_file,
    conda:
        "envs/python.yaml"
    shell:
        "python scripts/merge_metrics.py -i {input} -o {output}"


rule compute_emitter_metrics:
    input:
        p=config.spots_template,
        t=config.sim_emitter_template,
    output:
        config.emitter_metrics_template,
    conda:
        "envs/python.yaml"
    shell:
        "python scripts/compute_emitter_metrics.py "
        "-p {input.p} "
        "-t {input.t} "
        "-o {output} "
        "--decoder {wildcards.decoder} "
        "--replicate {wildcards.replicate} "
        "--run {wildcards.run} "
        "--score {wildcards.score} "


rule plot_emitter_metrics:
    input:
        expand(
            config.emitter_metrics_template,
            decoder=config.decoders,
            allow_missing=True,
        ),
    output:
        config.emitter_metrics_plot_template,
    params:
        " ".join(config.decoders),
    conda:
        "envs/python.yaml"
    shell:
        "python scripts/plot_emitter_metrics.py -d {params} -i {input} -o {output}"


rule merge_emitter_metrics:
    input:
        expand(
            config.emitter_metrics_template,
            decoder=config.decoders,
            replicate=range(config.num_replicates),
            run=config.runs,
            score=config.scores,
        ),
    output:
        config.emitter_metrics_file,
    conda:
        "envs/python.yaml"
    shell:
        "python scripts/merge_metrics.py -i {input} -o {output}"


rule compute_summary_metrics:
    input:
        b=config.bulk_metrics_template,
        e=config.emitter_metrics_template,
    output:
        config.summary_metrics_template,
    conda:
        "envs/python.yaml"
    shell:
        "python scripts/compute_summary_metrics.py -b {input.b} -e {input.e} -o {output}"


rule merge_summary_metrics:
    input:
        expand(
            config.summary_metrics_template,
            decoder=config.decoders,
            replicate=range(config.num_replicates),
            run=config.runs,
            score=config.scores,
        ),
    output:
        config.summary_metrics_file,
    conda:
        "envs/python.yaml"
    shell:
        "python scripts/merge_metrics.py -i {input} -o {output}"
