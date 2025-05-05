import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from ast import literal_eval
from hashlib import sha256


def load_data_2024_10_10(data_path: str,
                         ignore_hash: bool = False):
    expected_hash = "01d513fe3165bd5eb3ea6a9be52dc54afcf3d7f06cd398c18979653812797d18"  # intentional hardcoding
    if _get_file_hash(data_path) != expected_hash and not ignore_hash:
        raise ValueError("Data differs from expected; verify that you have the correct file.")
    data = pd.read_excel(data_path, skiprows=2, index_col="Pt ID (LAB)")
    return data


def _get_file_hash(file_path):
    with open(file_path, "rb") as f:
        file_hash = sha256(f.read()).hexdigest()
    return file_hash


def load_data(data_path: str,
              required_features: list = None,
              features_to_onehot_encode: list = None,
              features_to_drop: list = ("Column1", "Column2", "Column3", "APOE genotype", "APOE4 presence"),
              stratification_features: list = None) -> (pd.DataFrame, list):
    """
    Loads the data file. WARNING: This function is specific to the data file as-given for the project. It will not
    generalize since there are multiple operations specific to that data file.
    Parameters
    ----------
    data_path : str
        Path to the data.
    required_features : list
        Features that must be defined (not NaN); observations without these features will be dropped.
    features_to_onehot_encode : list
        Features to convert from their format to a one-hot encoding.
    features_to_drop : list
        Features to remove from the dataset. Generally intended for blank features.

    Returns
    -------
    pd.DataFrame
        Loaded data with appropriate subjects dropped, features encoded, and features dropped.
    list
        Updated stratification feature list with onehot-encoded names.
    """

    data = pd.read_excel(data_path, index_col="Pt", skiprows=2)

    if required_features is None:
        required_features = []
    if features_to_onehot_encode is None:
        features_to_onehot_encode = []
    if features_to_drop is None:
        features_to_drop = []
    if stratification_features is None:
        stratification_features = []

    # Remove extraneous rows
    s = [bool(a) for a in data.index.isna()]
    s = [not (a) and b != "average" for (a, b) in zip(s, data.index)]
    data = data.loc[s]

    # Some entries have things that they shouldn't; fix
    columns = data.columns
    cols_to_convert = set()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if isinstance(data.iloc[i, j], str):
                if data.iloc[i, j].endswith("*"):  # some entries end in *
                    data.iloc[i, j] = float(data.iloc[i, j][:-1])
                    cols_to_convert.add(columns[j])
                elif data.iloc[i, j] == "na":  # some entries are "na" instead of blank
                    data.iloc[i, j] = 0
                    cols_to_convert.add(columns[j])
                elif "," in data.iloc[i, j]:  # some entries use "," as the floating point
                    data.iloc[i, j] = literal_eval(data.iloc[i, j].replace(",", "."))
                    cols_to_convert.add(columns[j])
    for col in cols_to_convert:
        try:
            data[col] = data[col].astype(float)
        except Exception:
            continue

    # Remove observations that don't have defined values for the required features
    to_drop = [False for _ in range(data.shape[0])]
    for f in required_features:
        to_drop |= data[f].isna().values
    data = data.loc[~to_drop, :]

    for feat in features_to_onehot_encode:
        ohe = OneHotEncoder(sparse_output=False)
        recoded = ohe.fit_transform(data[[feat]])
        data.loc[:, ohe.get_feature_names_out()] = recoded
        if feat in stratification_features:
            stratification_features.pop(stratification_features.index(feat))
            stratification_features += list(ohe.get_feature_names_out())
        data.drop(feat, axis=1, inplace=True)

    for feat in features_to_drop:
        data.drop(feat, axis=1, inplace=True)
    return data, stratification_features


def extract_feature_groups(data):
    """Extracts features into groups. Function does not generalize."""
    demographics = list(data.columns[3:13]) + list(data.columns[-7:])
    demographics = demographics[0:1] + demographics[-7:]
    cognition = list(data.columns[13:16])
    brain_pathology = list(data.columns[16:39])
    cp_features = ["Retinal Cp % area", "Brain Cp % area"]
    retinal_biomarkers = list(data.columns[39:-7])
    disease_measures = cognition + ["Braak Stage", "ABC sum", "ABC average"]
    for c in cp_features:
        retinal_biomarkers.pop(retinal_biomarkers.index(c))

    return retinal_biomarkers, brain_pathology, demographics, cognition, cp_features, disease_measures
