import pandas as pd
def _parse_maf_to_spmg_format(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['Project', 'Sample', 'ID', 'Genome', 'mut_type', 'chrom', 'pos_start', 'pos_end', 'ref', 'alt', 'Type']
    if 'Tumor_Sample_Barcode' not in df.columns:
        df['Tumor_Sample_Barcode'] = 'Sample1'
    res = pd.DataFrame()
    res['Project'] = ['Project'] * len(df)
    res['Sample'] = df['Tumor_Sample_Barcode']
    res['ID'] = ['ID'] * len(df)
    res['Genome'] = ['GRCh37'] * len(df)
    res['mut_type'] = ['SNP'] * len(df)
    res['chrom'] = df['Chromosome']
    res['pos_start'] = df['Start_Position']
    res['pos_end'] = df['End_Position']
    res['ref'] = df['Reference_Allele']
    res['alt'] = df['Tumor_Seq_Allele2']
    res['Type'] = ['SOMATIC'] * len(df)
    return res