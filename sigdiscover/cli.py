import os
import sys
import click
import pandas as pd
import numpy as np
from sigdiscover.utils.logging import logger
from sigdiscover.utils.io import ensure_dir, save_matrix, load_matrix
from sigdiscover.config import Config
from sigdiscover.data.download import download_cosmic_signatures, download_tcga_mutations
from sigdiscover.data.loaders import load_maf, load_cosmic_signatures
from sigdiscover.matrices.sbs import build_sbs96_matrix
from sigdiscover.matrices.dbs import build_dbs78_matrix
from sigdiscover.matrices.id import build_id83_matrix
from sigdiscover.extraction.nmf import nmf_mutational_signatures
from sigdiscover.extraction.rank_selection import select_optimal_rank
from sigdiscover.assignment.similarity import assign_to_cosmic
from sigdiscover.validation.benchmark import run_benchmark_suite
from sigdiscover.validation.report import generate_validation_report
from sigdiscover.visualization.signature_plots import plot_signature_profile
from sigdiscover.visualization.activity_plots import plot_activity_barplot, plot_activity_heatmap
from sigdiscover.visualization.comparative import plot_cosine_heatmap, plot_rank_selection

@click.group()
def cli(): pass

@cli.command()
@click.option('--version', default='3.4')
@click.option('--output', default='data/cosmic/')
def download_cosmic(version, output):
    download_cosmic_signatures(version=version, output_dir=output)

@cli.command()
@click.option('--project', default='TCGA-BRCA')
@click.option('--output', default='data/tcga/')
def download_tcga(project, output):
    download_tcga_mutations(project=project, output_dir=output)

@cli.command()
@click.option('--input', 'input_file', required=True)
@click.option('--output', required=True)
@click.option('--type', 'matrix_type', default='SBS96')
def build_matrix(input_file, output, matrix_type):
    df = load_maf(input_file)
    matrix = build_sbs96_matrix(df) if matrix_type == 'SBS96' else (build_dbs78_matrix(df) if matrix_type == 'DBS78' else build_id83_matrix(df))
    save_matrix(matrix, os.path.join(output, f"{matrix_type}.tsv"))

@cli.command()
@click.option('--matrix', required=True)
@click.option('--output', required=True)
@click.option('--min-k', default=1)
@click.option('--max-k', default=10)
@click.option('--replicates', default=100)
def extract(matrix, output, min_k, max_k, replicates):
    ensure_dir(output)
    M = load_matrix(matrix).values
    res = select_optimal_rank(M, min_k=min_k, max_k=max_k, n_replicates=replicates)
    S, A, _ = nmf_mutational_signatures(M, n_signatures=res['optimal_k'])
    pd.DataFrame(S).to_csv(os.path.join(output, "signatures.tsv"), sep='\t')
    pd.DataFrame(A).to_csv(os.path.join(output, "activities.tsv"), sep='\t')
    res['all_k_results'].to_csv(os.path.join(output, "rank_selection.tsv"), sep='\t')

@cli.command()
@click.option('--signatures', required=True)
@click.option('--cosmic', required=True)
@click.option('--output', required=True)
def assign(signatures, cosmic, output):
    ensure_dir(output)
    S_disc = pd.read_csv(signatures, sep='\t', index_col=0).values
    S_cosmic, names = load_cosmic_signatures("SBS96", cosmic)
    assignments = assign_to_cosmic(S_disc, S_cosmic, names)
    assignments.to_csv(os.path.join(output, "assignments.tsv"), sep='\t', index=False)

@cli.command()
@click.option('--maf')
@click.option('--synthetic', is_flag=True)
@click.option('--output', required=True)
@click.option('--config', 'config_file')
def run(maf, synthetic, output, config_file):
    ensure_dir(output)
    cfg = Config.from_yaml(config_file) if config_file else Config()
    if synthetic:
        from sigdiscover.extraction.simulation import simulate_mutations
        muts = simulate_mutations(np.random.dirichlet(np.ones(96), size=3), np.random.uniform(10, 1000, size=(50, 3)))
        M_df = build_sbs96_matrix(muts)
    else:
        M_df = build_sbs96_matrix(load_maf(maf))
    M = M_df.values
    res = select_optimal_rank(M, min_k=cfg.extraction.min_signatures, max_k=cfg.extraction.max_signatures, n_replicates=5)
    S, A, _ = nmf_mutational_signatures(M, n_signatures=res['optimal_k'])
    save_matrix(pd.DataFrame(S), os.path.join(output, "signatures.tsv"))
    save_matrix(pd.DataFrame(A, index=M_df.index), os.path.join(output, "activities.tsv"))
    plot_rank_selection(res['all_k_results'], os.path.join(output, "rank_selection.png"))
    plot_activity_heatmap(pd.DataFrame(A, index=M_df.index), os.path.join(output, "activity_heatmap.png"))
    plot_activity_barplot(pd.DataFrame(A, index=M_df.index), save_path=os.path.join(output, "activity_barplot.png"))
    for i in range(res['optimal_k']): plot_signature_profile(S[i], title=f"Signature {i+1}", save_path=os.path.join(output, f"sig_{i+1}.png"))

@cli.command()
@click.option('--results', required=True)
@click.option('--output', required=True)
def report(results, output):
    import json
    bm_files = [f for f in os.listdir(results) if f.startswith("benchmark") and f.endswith(".json")]
    bm_data = json.load(open(os.path.join(results, bm_files[0]))) if bm_files else {"timestamp": "N/A", "benchmarks": {"synthetic_recovery": {"precision": 0, "recall": 0, "f1": 0, "mean_cosine": 0, "exposure_r": 0}, "cross_validation": {"test_reconstruction_r2": 0}, "noise_stability": {"0.1": 0}, "sigprofiler_agreement": 0}}
    generate_validation_report(bm_data, output)

@cli.command()
@click.option('--config', 'config_file')
@click.option('--output', required=True)
def benchmark(config_file, output):
    import yaml
    run_benchmark_suite(yaml.safe_load(open(config_file)) if config_file else {}, output)