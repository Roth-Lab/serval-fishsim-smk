from utils import ConfigManager

cfg = ConfigManager(config)


rule all:
    input:
        cfg.pipeline_files,


rule simulate_data:
    input:
        b=cfg.codebook_file,
        c=cfg.run_config_template,
        d=cfg.data_org_file,
    output:
        e=cfg.sim_emitter_template,
        i=cfg.sim_img_template,
    conda:
        "envs/fishsim.yaml"
    shell:
        "fishsim simulate -b {input.b} -c {input.c} -d {input.d} -e {output.e} -i {output.i} --seed {wildcards.replicate}"


rule serval_codebook:
    input:
        c=cfg.codebook_file,
        d=cfg.data_org_file,
    output:
        cfg.serval_codebook_file,
    conda:
        "envs/python.yaml"
    shell:
        "python scripts/convert_codebook_to_serval_codebook.py -c {input.c} -d {input.d} -o {output}"


rule run_savannah:
    input:
        c=cfg.codebook_file,
        i=cfg.sim_img_template,
    output:
        cfg.spots_template,
    wildcard_constraints:
        decoder="savannah",
    threads: cfg.savannah_threads
    conda:
        "envs/savannah.yaml"
    shell:
        "python scripts/run_savannah.py -c {input.c} -i {input.i} -o {output} -t {threads}"
        

rule run_bardensr:
    input:
        c=cfg.codebook_file,
        i=cfg.sim_img_template,
    output:
        cfg.spots_template,
    wildcard_constraints:
        decoder="bardensr",
    threads: cfg.bardensr_threads
    conda:
        "envs/bardensr.yaml"
    shell:
        "python scripts/run_bardensr.py -c {input.c} -i {input.i} -o {output} -t {threads}"


rule run_deepcell:
    input:
        c=cfg.codebook_file,
        d=cfg.data_org_file,
        i=cfg.sim_img_template,
    output:
        cfg.spots_template,
    params:
        cfg.deepcell_model_path,
    wildcard_constraints:
        decoder="deepcell",
    conda:
        "envs/deepcell.yaml"
    threads: cfg.deepcell_threads
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
        cfg.jsit_psf_file,
    params:
        j=cfg.jsit_src_dir,
        p=cfg.jsit_patch_size,
        s=cfg.jsit_scale_factor,
    conda:
        "envs/jsit.yaml"
    shell:
        "python scripts/build_jsit_psf.py -j {params.j} -p {params.p} -s {params.s} -o {output}"


rule preprocess_imgs:
    input:
        cfg.sim_img_template,
    output:
        cfg.sim_pp_img_template,
    conda:
        "envs/serval.yaml"
    shell:
        "python scripts/preprocess.py -i {input} -o {output}"


rule run_jsit:
    input:
        c=cfg.codebook_file,
        i=cfg.sim_pp_img_template,
        p=cfg.jsit_psf_file,
    output:
        cfg.jsit_spots_template,
    params:
        j=cfg.jsit_src_dir,
        p=cfg.jsit_patch_size,
        s=cfg.jsit_scale_factor,
    # wildcard_constraints:
    #     decoder="jsit",
    conda:
        "envs/jsit.yaml"
    threads: cfg.jsit_threads
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
        p=cfg.jsit_spots_template,
        t=cfg.sim_emitter_template,
    output:
        cfg.jsit_emitter_metrics_template,
    conda:
        "envs/python.yaml"
    shell:
        "python scripts/compute_emitter_metrics.py "
        "-p {input.p} "
        "-t {input.t} "
        "-o {output} "
        "--decoder jsit-{wildcards.jsit_penalty}-{wildcards.jsit_threshold} "
        "--nn-dist 1 "
        "--replicate {wildcards.replicate} "
        "--run {wildcards.run}"


rule select_best_jsit:
    input:
        e=expand(
            cfg.jsit_emitter_metrics_template,
            jsit_penalty=cfg.jsit_penalties,
            jsit_threshold=cfg.jsit_thresholds,
            allow_missing=True,
        ),
        s=expand(
            cfg.jsit_spots_template,
            jsit_penalty=cfg.jsit_penalties,
            jsit_threshold=cfg.jsit_thresholds,
            allow_missing=True,
        ),
    output:
        cfg.spots_template,
    wildcard_constraints:
        decoder="jsit",
    conda:
        "envs/python.yaml"
    shell:
        "python scripts/select_best_jsit_run.py -e {input.e} -s {input.s} -o {output}"


rule run_serval:
    input:
        c=cfg.serval_codebook_file,
        i=cfg.sim_img_template,
    output:
        cfg.spots_template,
    params:
        cfg.get_decoder_args,
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
        p=cfg.spots_template,
        t=cfg.sim_emitter_template,
    output:
        cfg.bulk_metrics_template,
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
        expand(cfg.bulk_metrics_template, decoder=cfg.decoders, allow_missing=True),
    output:
        cfg.bulk_metrics_plot_template,
    params:
        " ".join(cfg.decoders),
    conda:
        "envs/python.yaml"
    shell:
        "python scripts/plot_bulk_metrics.py -d {params} -i {input} -o {output}"


rule merge_bulk_metrics:
    input:
        expand(
            cfg.bulk_metrics_template,
            decoder=cfg.decoders,
            replicate=range(cfg.num_replicates),
            run=cfg.runs,
        ),
    output:
        cfg.bulk_metrics_file,
    conda:
        "envs/python.yaml"
    shell:
        "python scripts/merge_metrics.py -i {input} -o {output}"


rule compute_emitter_metrics:
    input:
        p=cfg.spots_template,
        t=cfg.sim_emitter_template,
    output:
        cfg.emitter_metrics_template,
    conda:
        "envs/python.yaml"
    shell:
        "python scripts/compute_emitter_metrics.py "
        "-p {input.p} "
        "-t {input.t} "
        "-o {output} "
        "--decoder {wildcards.decoder} "
        "--nn-dist 1 "
        "--replicate {wildcards.replicate} "
        "--run {wildcards.run}"


rule plot_emitter_metrics:
    input:
        expand(
            cfg.emitter_metrics_template,
            decoder=cfg.decoders,
            allow_missing=True,
        ),
    output:
        cfg.emitter_metrics_plot_template,
    params:
        " ".join(cfg.decoders),
    conda:
        "envs/python.yaml"
    shell:
        "python scripts/plot_emitter_metrics.py -d {params} -i {input} -o {output}"


rule merge_emitter_metrics:
    input:
        expand(
            cfg.emitter_metrics_template,
            decoder=cfg.decoders,
            replicate=range(cfg.num_replicates),
            run=cfg.runs,
        ),
    output:
        cfg.emitter_metrics_file,
    conda:
        "envs/python.yaml"
    shell:
        "python scripts/merge_metrics.py -i {input} -o {output}"


rule compute_summary_metrics:
    input:
        b=cfg.bulk_metrics_template,
        e=cfg.emitter_metrics_template,
    output:
        cfg.summary_metrics_template,
    conda:
        "envs/python.yaml"
    shell:
        "python scripts/compute_summary_metrics.py -b {input.b} -e {input.e} -o {output}"


rule merge_summary_metrics:
    input:
        expand(
            cfg.summary_metrics_template,
            decoder=cfg.decoders,
            replicate=range(cfg.num_replicates),
            run=cfg.runs,
        ),
    output:
        cfg.summary_metrics_file,
    conda:
        "envs/python.yaml"
    shell:
        "python scripts/merge_metrics.py -i {input} -o {output}"
