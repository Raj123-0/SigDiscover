import os
import tempfile
import pandas as pd
from sigdiscover.matrices.context import _parse_maf_to_spmg_format
from sigdiscover.utils.logging import logger
from sigdiscover.utils.io import ensure_dir

def _install_genome(genome: str):
    '''Installs the requested genome reference using SigProfilerMatrixGenerator.'''
    logger.info(f"Installing reference genome: {genome}")
    try:
        from SigProfilerMatrixGenerator import install
        install.install(genome)
    except ImportError:
        logger.error("SigProfilerMatrixGenerator is not installed. Cannot install genome references.")
        raise RuntimeError(f"Cannot install genome {genome} without SigProfilerMatrixGenerator.")
    except Exception as e:
        logger.error(f"Failed to install genome {genome}: {e}")
        raise RuntimeError(f"Failed to install genome {genome}: {e}")

def _get_revcomp(seq: str) -> str:
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return "".join(complement.get(base, 'N') for base in reversed(seq))

def _fallback_sbs96_matrix(mutations: pd.DataFrame, genome: str) -> pd.DataFrame:
    '''Fallback context extraction using pysam to read the fasta file.'''
    import pysam
    logger.warning("SigProfilerMatrixGenerator not available. Using fallback context extraction via pysam.")
    fasta_dir = "data/reference"
    ensure_dir(fasta_dir)
    fasta_path = os.path.join(fasta_dir, f"{genome}.fa.gz")

    if not os.path.exists(fasta_path):
        _install_genome(genome)

    bases = ['A', 'C', 'G', 'T']
    subs = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    cols = [f"{five}[{sub}]{three}" for sub in subs for five in bases for three in bases]

    samples = mutations['Tumor_Sample_Barcode'].unique() if 'Tumor_Sample_Barcode' in mutations.columns else []
    df = pd.DataFrame(0, index=samples, columns=cols)

    if not os.path.exists(fasta_path):
        raise RuntimeError(f"Reference genome not found at {fasta_path}. Cannot generate SBS96.")

    try:
        fasta = pysam.FastaFile(fasta_path)
    except ValueError as e:
        raise RuntimeError(f"Failed to open reference genome with pysam: {e}")

    for _, row in mutations.iterrows():
        sample = row.get('Tumor_Sample_Barcode')
        ref = row.get('Reference_Allele')
        alt = row.get('Tumor_Seq_Allele2')
        chrom = str(row.get('Chromosome'))
        if not chrom.startswith('chr'):
            chrom = f"chr{chrom}"

        try:
            start = int(row.get('Start_Position'))
        except (ValueError, TypeError):
            continue

        if ref not in bases or alt not in bases or ref == alt:
            continue

        try:
            context = fasta.fetch(chrom, start - 2, start + 1).upper()
        except Exception:
            continue

        if len(context) != 3 or context[1] != ref:
            continue

        up, mid, down = context[0], context[1], context[2]

        if mid in ('A', 'G'):
            up, mid, down = _get_revcomp(down), _get_revcomp(mid), _get_revcomp(up)
            alt = _get_revcomp(alt)

        sub = f"{mid}>{alt}"
        col = f"{up}[{sub}]{down}"

        if col in df.columns:
            df.at[sample, col] += 1

    return df

def build_sbs96_matrix(mutations: pd.DataFrame, genome: str = "GRCh37") -> pd.DataFrame:
    if len(mutations) == 0:
        return _fallback_sbs96_matrix(mutations, genome)
    try:
        from SigProfilerMatrixGenerator.scripts import SigProfilerMatrixGeneratorFunc as spmg
    except ImportError:
        return _fallback_sbs96_matrix(mutations, genome)

    spmg_df = _parse_maf_to_spmg_format(mutations, genome)
    spmg_df = spmg_df[spmg_df['ref'].str.len() == 1]
    spmg_df = spmg_df[spmg_df['alt'].str.len() == 1]
    if len(spmg_df) == 0:
        return _fallback_sbs96_matrix(mutations, genome)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        os.makedirs(input_dir)
        input_file = os.path.join(input_dir, "input.txt")
        spmg_df.to_csv(input_file, sep='\t', index=False)
        try:
            spmg.SigProfilerMatrixGeneratorFunc("test", genome, input_dir, plot=False, exome=False, bed_file=None, chrom_based=False, tsb_stat=False, seqInfo=False, cushion=100)
        except Exception as e:
            if "not installed" in str(e).lower() or "reference" in str(e).lower():
                _install_genome(genome)
                try:
                    spmg.SigProfilerMatrixGeneratorFunc("test", genome, input_dir, plot=False, exome=False, bed_file=None, chrom_based=False, tsb_stat=False, seqInfo=False, cushion=100)
                except Exception:
                    return _fallback_sbs96_matrix(mutations, genome)
            else:
                return _fallback_sbs96_matrix(mutations, genome)
        matrix_file = os.path.join(input_dir, "output", "SBS", "test.SBS96.all")
        if os.path.exists(matrix_file):
            df = pd.read_csv(matrix_file, sep='\t', index_col=0)
            return df.T
        return _fallback_sbs96_matrix(mutations, genome)
