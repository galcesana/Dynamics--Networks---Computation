import pandas as pd

# =============================================================================
# 1. Load BK-dict.txt  → columns: (lnum, abbr, tnum, termFull)
#    (Tab-delimited; no header.)
# =============================================================================
bk_dict_df = pd.read_csv(
    './BK-dict.txt',
    sep=r'\s+',
    header=None,
    names=['lnum', 'termAbbr', 'tnum', 'termFull', 'comment'],
    encoding='utf-8',
    comment='#'
)
# Drop the comment column if it exists
bk_dict_df = bk_dict_df.drop('comment', axis=1, errors='ignore')

# =============================================================================
# 2. Load BK-term.txt  → columns: (lnum, spkrID, chipID, termAbbr)
#    (Whitespace-delimited; no header.)
# =============================================================================
bk_term_df = pd.read_csv(
    './BK-term.txt',
    sep=r'\s+',
    header=None,
    names=['lnum', 'spkrID', 'chipID', 'termAbbr']
)

# =============================================================================
# 3. Load cnum-vhcm-lab-new.txt  → columns: (chipID, WCS_hue, WCS_value, WCS_chroma, CIEL_L, CIEL_a, CIEL_b)
#    (Whitespace-delimited; no header.)
# =============================================================================
lab_df = pd.read_csv(
    './cnum-vhcma-lab-new.txt',
    sep=r'\s+',
    header=None,
    names=['chipID', 'WCS_hue', 'WCS_value', 'WCS_chroma', 'CIEL_L', 'CIEL_a', 'CIEL_b'],
    comment='#'
)

# =============================================================================
# 4. Merge bk_term_df with bk_dict_df to attach full orthographic termFull
#    for each (lnum, spkrID, chipID, termAbbr).
# =============================================================================
term_full_df = bk_term_df.merge(
    bk_dict_df[['lnum', 'termAbbr', 'termFull']],
    on=['lnum', 'termAbbr'],
    how='left'
)
# If any termFull is NaN, check for mismatches in abbreviations or whitespace.

# =============================================================================
# 5. Merge term_full_df with lab_df on chipID → to bring in (CIEL_L, CIEL_a, CIEL_b).
# =============================================================================
# Convert chipID to string in both dataframes
term_full_df['chipID'] = term_full_df['chipID'].astype(str)
lab_df['chipID'] = lab_df['chipID'].astype(str)

term_lab_df = term_full_df.merge(
    lab_df[['chipID', 'CIEL_L', 'CIEL_a', 'CIEL_b']],
    on='chipID',
    how='inner'
)

# =============================================================================
# 6. Map each lnum → its actual language name using a hard‐coded dict.
#    (Adjust the dictionary to match Appendix 1 EXACTLY.)
# =============================================================================
lnum_to_name = {
    1:  "Arabic",
    2:  "Bulgarian",
    3:  "Catalan",
    4:  "Cantonese",
    5:  "Danish",
    6:  "English",
    7:  "Hebrew",
    8:  "Hungarian",
    9:  "Ibibio",
    10: "Japanese",
    11: "Korean",
    12: "Mandarin",
    13: "Spanish",
    14: "Tagalog",
    15: "Swahili",
    16: "Tagalog",
    17: "Thai",
    18: "Tzeltal",
    19: "Urdu",
    20: "Vietnamese"
}

# Create a new column 'language' by mapping lnum → language name
term_lab_df['language'] = term_lab_df['lnum'].map(lnum_to_name)

# =============================================================================
# 7. Select only the columns we want in the final table:
#    - 'language'
#    - 'termFull'  → rename to 'term'
#    - 'CIEL_L', 'CIEL_a', 'CIEL_b'
# =============================================================================
final_df = term_lab_df[[
    'language',
    'termFull',
    'CIEL_L',
    'CIEL_a',
    'CIEL_b'
]].rename(columns={'termFull': 'term'})

# Now final_df has one row per (speaker, chip) choice:
#    language  | term       | CIEL_L | CIEL_a | CIEL_b
#    "Spanish" | "rojo"     | 50.12  |  68.45 | 63.22
#    "Spanish" | "amarillo" | 80.23  |  -1.45 | -25.77
#    … etc.

# =============================================================================
# 8. (Optional) If you want one row PER DISTINCT (language, term) combination—
#    you can drop duplicates on (language, term, CIEL_L, CIEL_a, CIEL_b).
# =============================================================================
unique_term_df = final_df.drop_duplicates(subset=['language', 'term', 'CIEL_L', 'CIEL_a', 'CIEL_b'])

# =============================================================================
# 9. Save both versions to CSV:
# =============================================================================
final_df.to_csv('./bk_merged_full.csv', index=False, encoding='utf-8')
unique_term_df.to_csv('./bk_merged_unique_terms.csv', index=False, encoding='utf-8')

print("Done! → Wrote:\n  • ./bk_merged_full.csv       (per speaker‐chip)\n  • ./bk_merged_unique_terms.csv (per language‐term)")
