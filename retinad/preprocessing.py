import pandas as pd


def compute_missingness(df):
    return df.isnull().sum() / len(df)


def extract_diagnosis(df: pd.DataFrame):
    """Extracts diagnosis from dataframe index"""
    diag_list = []
    for id in df.index:
        if id.startswith("NC"):
            diag_list.append("NC")
        elif id.startswith("MCI"):
            diag_list.append("MCI")
        elif id.startswith("AD"):
            diag_list.append("AD")
    df["diagnosis"] = diag_list
    return


def _extract_feature_groups(df):
    """Extracts retinal, brain pathology, and cognitive measures. Note that this applies to a specific dataset and
    should not be used on other datasets without first verifying that the extraction works as expected."""
    retinal_features = [c for c in df.columns if "retinal" in c.lower()]
    retinal_features += list(df.columns[-2:])

    brain_pathology_features = [c for c in df.columns if "brain" in c.lower()]
    brain_pathology_features += ["Braak Stage", "ABC average", "CAA score"]

    cognitive_features = ["CDR Global [score]", "MMSE [score]", "MOCA"]

    # Make sure that there isn't any overlap between these
    assert len(set(retinal_features).intersection(set(brain_pathology_features))) == 0
    assert len(set(retinal_features).intersection(set(cognitive_features))) == 0
    assert len(set(cognitive_features).intersection(set(brain_pathology_features))) == 0

    return retinal_features, brain_pathology_features, cognitive_features


def compute_missingness_by_category(df, column):
    return df.groupby(column).apply(lambda x: compute_missingness(x))
