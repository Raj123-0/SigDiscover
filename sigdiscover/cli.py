import os
import click
import numpy as np
from sigdiscover.utils.io import ensure_dir, save_matrix, load_matrix
from sigdiscover.config import Config
from sigdiscover.data.download import download_cosmic_signatures, download_tcga_mutations
from sigdiscover.data.loaders import load_cosmic_signatures, load_maf
from sigdiscover.data.validators import validate_matrix
from sigdiscover.extraction.nmf import nmf_mutational_signatures
from sigdiscover.extraction.rank_selection import select_optimal_rank
from sigdiscover.matrices.dbs import build_dbs78_matrix
from sigdiscover.matrices.id import build_id83_matrix
from sigdiscover.matrices.sbs import build_sbs96_matrix
from sigdiscover.utils.io import (
    ensure_dir,
    load_matrix,
    save_activities,
    save_matrix,
    save_signatures,
)
from sigdiscover.utils.logging import logger
from sigdiscover.validation.benchmark import run_benchmark_suite
from sigdiscover.validation.report import generate_validation_report
from sigdiscover.visualization.activity_plots import plot_activity_barplot, plot_activity_heatmap
from sigdiscover.visualization.comparative import plot_rank_selection

@click.group()
def cli(): pass

@cli.command()
@click.option('--version', default='3.4', type=str)
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
@click.option('--replicates', default=None)
@click.option('--config', 'config_file')
def extract(matrix, output, min_k, max_k, replicates, config_file):
    ensure_dir(output)
    cfg = Config.from_yaml(config_file) if config_file else Config()

    # CLI args override config if provided
    final_min_k = min_k if min_k != 1 else cfg.extraction.min_signatures
    final_max_k = max_k if max_k != 10 else cfg.extraction.max_signatures
    final_reps = int(replicates) if replicates is not None else cfg.extraction.n_replicates

    M_df = load_matrix(matrix)
    val_report = validate_matrix(M_df)
    if not val_report['is_valid']:
        logger.warning(f"Matrix validation warnings/errors: {val_report['errors']}")
    M = M_df.values
    res = select_optimal_rank(M, min_k=final_min_k, max_k=final_max_k, n_replicates=final_reps)
    S, A, _ = nmf_mutational_signatures(
        M,
        n_signatures=res['optimal_k'],
        n_iterations=cfg.extraction.n_iterations,
        tolerance=cfg.extraction.tolerance,
        init_method=cfg.extraction.init_method
    )
    channels = list(load_matrix(matrix).columns)
    save_signatures(pd.DataFrame(S), os.path.join(output, "signatures.tsv"), channels=channels)
    save_activities(pd.DataFrame(A), os.path.join(output, "activities.tsv"))
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
@click.option('--skip-assignment', is_flag=True, help='Skip COSMIC assignment')
def run(maf, synthetic, output, config_file, skip_assignment):
    if bool(maf) == bool(synthetic):
        raise click.UsageError("Specify exactly one of --maf or --synthetic")
    ensure_dir(output)
    cfg = Config.from_yaml(config_file) if config_file else Config()
    if synthetic:
        from sigdiscover.extraction.simulation import simulate_mutations
        rng = np.random.default_rng(cfg.project.seed)
        muts = simulate_mutations(rng.dirichlet(np.ones(96), size=3), rng.uniform(10, 1000, size=(50, 3)), seed=cfg.project.seed)
        M_df = build_sbs96_matrix(muts)
    else:
        M_df = build_sbs96_matrix(load_maf(maf))
    val_report = validate_matrix(M_df)
    if not val_report['is_valid']:
        logger.warning(f"Matrix validation warnings/errors: {val_report['errors']}")
    M = M_df.values
    res = select_optimal_rank(M, min_k=cfg.extraction.min_signatures, max_k=cfg.extraction.max_signatures, n_replicates=cfg.extraction.n_replicates)
    S, A, _ = nmf_mutational_signatures(
        M,
        n_signatures=res['optimal_k'],
        n_iterations=cfg.extraction.n_iterations,
        tolerance=cfg.extraction.tolerance,
        init_method=cfg.extraction.init_method
    )
    save_signatures(pd.DataFrame(S), os.path.join(output, "signatures.tsv"), channels=list(M_df.columns))
    save_activities(pd.DataFrame(A, index=M_df.index), os.path.join(output, "activities.tsv"))
    plot_rank_selection(res['all_k_results'], os.path.join(output, "rank_selection.png"))
    plot_activity_heatmap(pd.DataFrame(A, index=M_df.index), os.path.join(output, "activity_heatmap.png"))
    plot_activity_barplot(pd.DataFrame(A, index=M_df.index), save_path=os.path.join(output, "activity_barplot.png"))
    for i in range(res['optimal_k']):
        plot_signature_profile(S[i], title=f"Signature {i+1}", save_path=os.path.join(output, f"sig_{i+1}.png"))

    if not skip_assignment:
        try:
            S_cosmic, names = load_cosmic_signatures("SBS96", "data/cosmic")
            assignments = assign_to_cosmic(S, S_cosmic, names)
            assignments.to_csv(os.path.join(output, "assignments.tsv"), sep='\t', index=False)

            # Plot cosine heatmap
            plot_cosine_heatmap(S, S_cosmic, names, save_path=os.path.join(output, "cosine_heatmap.png"))
        except FileNotFoundError:
            logger.info("COSMIC reference data not found at data/cosmic. Skipping assignment.")
        except Exception as e:
            logger.warning(f"Assignment failed: {e}")
    else:
        logger.info("Skipping COSMIC assignment due to --skip-assignment flag.")

@cli.command()
@click.option('--results', required=True)
@click.option('--output', required=True)
def report(results, output):
    import json
    if not os.path.isdir(results):
        raise click.UsageError(f"--results must be a directory. Got: {results}")
    bm_files = [f for f in os.listdir(results) if f.startswith("benchmark_") and f.endswith(".json")]
    if not bm_files:
        raise click.UsageError(f"No benchmark_*.json files found in {results}")

    def extract_time(f):
        try:
            return datetime.strptime(f, "benchmark_%Y%m%d_%H%M%S.json")
        except ValueError:
            return datetime.min

    from datetime import datetime
    latest_file = max(bm_files, key=extract_time)

    with open(os.path.join(results, latest_file)) as f:
        bm_data = json.load(f)

    generate_validation_report(bm_data, output)

@cli.command()
@click.option('--config', 'config_file')
@click.option('--output', required=True)
def benchmark(config_file, output):
    if config_file:
        cfg = Config.from_yaml(config_file)
        # convert to dict for run_benchmark_suite if it expects dict, or update benchmark suite to take Config.
        # Benchmark.py expects config as Dict, let's pass dict or modify benchmark.
        import dataclasses
        cfg_dict = dataclasses.asdict(cfg)
    else:
        import dataclasses
        cfg_dict = dataclasses.asdict(Config())
    run_benchmark_suite(cfg_dict, output)
