
import pandas as pd

def _determine_mut_type(ref, alt):
    if not isinstance(ref, str) or not isinstance(alt, str):
        return 'UNKNOWN'
    if len(ref) == 1 and len(alt) == 1:
        return 'SNP'
    elif len(ref) == 2 and len(alt) == 2:
        return 'DNP'
    elif len(ref) > len(alt):
        return 'DEL'
    elif len(ref) < len(alt):
        return 'INS'
    else:
        return 'UNKNOWN'

def _parse_maf_to_spmg_format(df: pd.DataFrame, genome: str = 'GRCh37') -> pd.DataFrame:

    # Do not mutate the caller's DataFrame
    input_df = df.copy()
    if 'Tumor_Sample_Barcode' not in input_df.columns:
        input_df['Tumor_Sample_Barcode'] = 'Sample1'

    res = pd.DataFrame()
    res['Project'] = ['Project'] * len(input_df)
    res['Sample'] = input_df['Tumor_Sample_Barcode']
    res['ID'] = ['ID'] * len(input_df)
    res['Genome'] = [genome] * len(input_df)

    res['chrom'] = input_df['Chromosome']
    res['pos_start'] = input_df['Start_Position']
    res['pos_end'] = input_df['End_Position']
    res['ref'] = input_df['Reference_Allele']
    res['alt'] = input_df['Tumor_Seq_Allele2']

    mut_types = [
        _determine_mut_type(r, a)
        for r, a in zip(res['ref'], res['alt'])
    ]
    res['mut_type'] = mut_types

    res['Type'] = ['SOMATIC'] * len(input_df)
    return res
