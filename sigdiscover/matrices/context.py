
import pandas as pd

from sigdiscover.utils.logging import logger


def to_spmg_format(df: pd.DataFrame, genome: str = "GRCh37", context: str = "SBS") -> pd.DataFrame:
    '''Convert MAF to SigProfilerMatrixGenerator format.

    Args:
        df: Input MAF DataFrame
        genome: The genome build (e.g. GRCh37)
        context: Context of the signatures to build (SBS, DBS, ID)

    Returns:
        DataFrame in SPMG input format.
    '''
    if 'Tumor_Sample_Barcode' not in df.columns:
        raise ValueError("MAF requires a 'Tumor_Sample_Barcode' column.")

    res = pd.DataFrame()
    res['Project'] = ['Project'] * len(df)
    res['Sample'] = df['Tumor_Sample_Barcode']
    res['ID'] = ['ID'] * len(df)
    res['Genome'] = [genome] * len(df)

    # Map variant types
    def map_mut_type(vt):
        if pd.isna(vt): return 'SNP'
        vt = vt.upper()
        if vt in ('SNP', 'SNV'): return 'SNP'
        if vt in ('DNP', 'TNP', 'ONP'): return 'DNP' # Map all multi-nucleotide to DNP, spmg will handle
        if vt in ('DEL', 'DELETION'): return 'DEL'
        if vt in ('INS', 'INSERTION'): return 'INS'
        return 'SNP'

    if 'Variant_Type' in df.columns:
        res['mut_type'] = df['Variant_Type'].apply(map_mut_type)
    else:
        res['mut_type'] = ['SNP'] * len(df)

    # Filter based on context
    if context == 'SBS':
        mask = res['mut_type'] == 'SNP'
    elif context == 'DBS':
        mask = res['mut_type'] == 'DNP'
    elif context == 'ID':
        mask = res['mut_type'].isin(['DEL', 'INS'])
    else:
        mask = pd.Series([True] * len(df))

    res = res[mask].copy()

    # We must filter df using the same mask
    df_filtered = df[mask].copy()

    if len(res) == 0:
        return res

    # Chromosome normalization
    def normalize_chr(c):
        c = str(c).strip()
        if not c.startswith('chr'):
            c = 'chr' + c
        if c == 'chr23': c = 'chrX'
        if c == 'chr24': c = 'chrY'
        if c == 'chrMT': c = 'chrM'
        return c

    res['chrom'] = df_filtered['Chromosome'].apply(normalize_chr)

    # Check for non-standard chromosomes
    standard_chrs = {f"chr{i}" for i in range(1, 23)}.union({'chrX', 'chrY', 'chrM'})
    non_standard = set(res['chrom']) - standard_chrs
    if non_standard:
        logger.warning(f"Found non-standard chromosome names: {non_standard}")

    res['pos_start'] = df_filtered['Start_Position']
    res['pos_end'] = df_filtered['End_Position']
    res['ref'] = df_filtered['Reference_Allele']
    res['alt'] = df_filtered['Tumor_Seq_Allele2']
    res['Type'] = ['SOMATIC'] * len(res)

    return res
