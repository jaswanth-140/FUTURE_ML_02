# file: data_prep.py

import re
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df[['Ticket Description', 'Ticket Type', 'Ticket Priority']].copy()
    df.dropna(subset=['Ticket Description', 'Ticket Type', 'Ticket Priority'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def clean_text(text):
    # Architecture A only: lowercase + strip punctuation (digits kept — error codes, order #s)
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_stratify_labels(df):
    # combine both targets into one label so the split respects their joint distribution
    strat = df['Ticket Type'].astype(str) + '_' + df['Ticket Priority'].astype(str)

    # collapse combos that appear only once into a single 'RARE' bucket,
    # since sklearn requires every stratify class to have at least 2 members
    counts = strat.value_counts()
    rare_labels = counts[counts < 2].index
    strat = strat.where(~strat.isin(rare_labels), 'RARE')

    # if the RARE bucket itself only has 1 member, drop it (still too small)
    rare_count = (strat == 'RARE').sum()
    if rare_count == 1:
        keep_mask = strat != 'RARE'
        df = df[keep_mask].reset_index(drop=True)
        strat = strat[keep_mask].reset_index(drop=True)

    return df, strat


def split_data(df, strat, test_size=0.2):
    df = df.copy()
    df['cleaned_text'] = df['Ticket Description'].apply(clean_text)

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=strat
    )

    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)
    return train_df, test_df


def load_and_prep_data(filepath, test_size=0.2):
    df = load_data(filepath)
    df, strat = build_stratify_labels(df)
    train_df, test_df = split_data(df, strat, test_size)

    target_names = ['Ticket Type', 'Ticket Priority']

    X_train_raw = train_df['Ticket Description']      # for Architecture B (BERT)
    X_test_raw = test_df['Ticket Description']
    X_train_clean = train_df['cleaned_text']           # for Architecture A (TF-IDF)
    X_test_clean = test_df['cleaned_text']
    y_train = train_df[target_names].values
    y_test = test_df[target_names].values

    return (X_train_raw, X_test_raw, X_train_clean, X_test_clean,
            y_train, y_test, target_names)