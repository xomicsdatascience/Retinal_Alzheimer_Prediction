import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from retinad.preprocessing import compute_missingness_by_category, extract_diagnosis
from scipy.stats import pearsonr, false_discovery_control


def feature_missingness_by_diagnosis(df: pd.DataFrame,
                                     save_path: str = None,
                                     diagnosis_column: str = "diagnosis"):
    """Creates a bar plot of missingness grouped by diagnosis"""
    if diagnosis_column not in df.columns:
        extract_diagnosis(df)
    missingness_by_diagnosis = compute_missingness_by_category(df, diagnosis_column)

    plot = missingness_by_diagnosis.T.plot(kind="bar", figsize=(12,6))
    plt.title("Missingness of features by diagnosis")
    plt.ylabel("Missingness")
    if save_path is not None:
        plt.savefig(save_path)
    return plot


def pairplot_by_diagnosis(df: pd.DataFrame,
                          feature_list: list = None,
                          save_path: str = None,
                          diagnosis_column: str = "diagnosis"):
    if diagnosis_column not in df.columns:
        extract_diagnosis(df)
    if feature_list is None:
        feature_list = df.columns
    p = sns.pairplot(df[feature_list + [diagnosis_column]], hue=diagnosis_column, corner=True)
    if save_path is not None:
        p.savefig(save_path)
    return p


def pairplot_by_diagnosis_with_corrinfo(df: pd.DataFrame,
                                        feature_list: list = None,
                                        save_path: str = None,
                                        diagnosis_column: str = "diagnosis"):
    p = pairplot_by_diagnosis(df, feature_list, save_path=None, diagnosis_column=diagnosis_column)
    if feature_list is None:
        feature_list = df.columns

    # Compute pearson_r, pvalues
    pvals = np.zeros((len(feature_list), len(feature_list)))
    rvals = np.zeros((len(feature_list), len(feature_list)))
    for i, j in zip(*np.tril_indices_from(p.axes, 1)):
        if p.axes[i,j] is None or i == j:
            continue
        ri_retain = ~df[feature_list[i]].isna()
        rj_retain = ~df[feature_list[j]].isna()
        retain = ri_retain & rj_retain
        r, pv = pearsonr(df.loc[retain, feature_list[i]], df.loc[retain, feature_list[j]])
        pvals[i,j] = pv
        rvals[i,j] = r
    pvals[pvals > 0] = false_discovery_control(pvals[pvals>0])  # bh correction

    # Need first loop to complete first since we need all values for BH correction
    for i, j in zip(*np.tril_indices_from(p.axes, 1)):
        if p.axes[i,j] is None or i == j:
            continue
        ri_retain = ~df[feature_list[i]].isna()
        rj_retain = ~df[feature_list[j]].isna()
        retain = ri_retain & rj_retain

        p.axes[i, j].annotate(f"r = {round(rvals[i, j], 3)}", (1, 0.15), xycoords="axes fraction", ha="right", va="bottom")
        p.axes[i, j].annotate(f"p_adj = {'{:,.3E}'.format(pvals[i, j])}", (1, 0.075), xycoords="axes fraction", ha="right", va="bottom")
        p.axes[i, j].annotate(f"n = {sum(retain)}", (1, 0), xycoords="axes fraction", ha="right", va="bottom")
    if save_path is not None:
        p.savefig(save_path)
    return p


def heatmap_with_threshold(df: pd.DataFrame,
                           feature_list: list,
                           threshold: float = 0,
                           feature_desc_for_title: str = "",
                           save_path: str = None):
    if feature_list is None:
        feature_list = df.columns

    fig, ax = plt.subplots(1,1,figsize=(12,12))
    p_heatmap = sns.heatmap(df[feature_list].corr() * (np.abs(df[feature_list].corr() > threshold)), ax=ax, cmap="viridis")

    if feature_desc_for_title == "":
        hmap_title = f"Correlation heatmap thresholded at {threshold}"
    else:
        hmap_title = f"Correlation heatmap of {feature_desc_for_title} thresholded at {threshold}"
    ax.set_title(hmap_title)

    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax