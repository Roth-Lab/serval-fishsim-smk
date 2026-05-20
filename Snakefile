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


rule run_bardensr:
    input:
        c=config.codebook_file,
        i=config.sim_img_template,
    output:
        config.spots_template,
    wildcard_constraints:
        decoder="bardensr",
    threads: config.bardensr_threads
    conda:
        "envs/bardensr.yaml"
    shell:
        "python scripts/run_bardensr.py -c {input.c} -i {input.i} -o {output} -t {threads}"


rule run_deepcell:
    input:
        c=config.codebook_file,
        d=config.data_org_file,
        i=config.sim_img_template,
    output:
        config.spots_template,
    params:
        config.deepcell_model_path,
    wildcard_constraints:
        decoder="deepcell",
    conda:
        "envs/deepcell.yaml"
    threads: config.deepcell_threads
    shell:
        "python scripts/run_deepcell.py "
        "-c {input.c} "
        "-d {input.d} "
        "-i {input.i} "
        "-m {params} "
        "-o {output} "
        "-t {threads}"


rule build_jsit_psf:
    output:
        config.jsit_psf_file,
    params:
        j=config.jsit_src_dir,
        p=config.jsit_patch_size,
        s=config.jsit_scale_factor,
    conda:
        "envs/jsit.yaml"
    shell:
        "python scripts/build_jsit_psf.py -j {params.j} -p {params.p} -s {params.s} -o {output}"


rule preprocess_imgs:
    input:
        config.sim_img_template,
    output:
        config.sim_pp_img_template,
    conda:
        "envs/serval.yaml"
    shell:
        "python scripts/preprocess.py -i {input} -o {output}"


rule run_jsit:
    input:
        c=config.codebook_file,
        i=config.sim_pp_img_template,
        p=config.jsit_psf_file,
    output:
        config.jsit_spots_template,
    params:
        j=config.jsit_src_dir,
        p=config.jsit_patch_size,
        s=config.jsit_scale_factor,
    # wildcard_constraints:
    #     decoder="jsit",
    conda:
        "envs/jsit.yaml"
    threads: config.jsit_threads
    shell:
        "python scripts/run_jsit.py "
        "-c {input.c} "
        "-i {input.i} "
        "-j {params.j} "
        "-p {input.p} "
        "-o {output} "
        "-t {threads} "
        "--patch-size {params.p} "
        "--penalty {wildcards.jsit_penalty} "
        "--scale-factor {params.s} "
        "--sparsity-threshold {wildcards.jsit_threshold}"


rule compute_jsit_emitter_metrics:
    input:
        p=config.jsit_spots_template,
        t=config.sim_emitter_template,
    output:
        config.jsit_emitter_metrics_template,
    conda:
        "envs/python.yaml"
    shell:
        "python scripts/compute_emitter_metrics.py "
        "-p {input.p} "
        "-t {input.t} "
        "-o {output} "
        "--decoder jsit-{wildcards.jsit_penalty}-{wildcards.jsit_threshold} "
        "--replicate {wildcards.replicate} "
        "--run {wildcards.run}"


rule select_best_jsit:
    input:
        e=expand(
            config.jsit_emitter_metrics_template,
            jsit_penalty=config.jsit_penalties,
            jsit_threshold=config.jsit_thresholds,
            allow_missing=True,
        ),
        s=expand(
            config.jsit_spots_template,
            jsit_penalty=config.jsit_penalties,
            jsit_threshold=config.jsit_thresholds,
            allow_missing=True,
        ),
    output:
        config.spots_template,
    wildcard_constraints:
        decoder="jsit",
    conda:
        "envs/python.yaml"
    shell:
        "python scripts/select_best_jsit_run.py -e {input.e} -s {input.s} -o {output}"


rule run_serval:
    input:
        c=config.serval_codebook_file,
        i=config.sim_img_template,
    output:
        config.spots_template,
    params:
        config.get_decoder_args,
    wildcard_constraints:
        decoder="cosine|cosine-np|nn|scaled",
    conda:
        "envs/serval.yaml"
    shell:
        "python scripts/run_serval.py "
        "-c {input.c} "
        "-i {input.i} "
        "-o {output} "
        "{params}"


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
        "--run {wildcards.run}"


rule plot_bulk_metrics:
    input:
        expand(config.bulk_metrics_template, decoder=config.decoders, allow_missing=True),
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
        "--run {wildcards.run}"


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
        ),
    output:
        config.summary_metrics_file,
    conda:
        "envs/python.yaml"
    shell:
        "python scripts/merge_metrics.py -i {input} -o {output}"
