import pandas as pd
import numpy as np
import shap
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, StratifiedKFold
from typing import List

from retinad.dataload import load_data, extract_feature_groups
from argparse import ArgumentParser

random_state = 1


def make_directory_unique(path: str) -> str:
    """If the path already exists, appends an index to make a unique path."""
    idx = 0
    while os.path.exists(path):
        dirname = os.path.basename(path)
        dirname = dirname + "{:05d}".format(idx)
        path = os.path.join(os.path.dirname(path), dirname)
    return path


def write_project_summary(var_dict: dict,
                          save_path: str):
    """Prints out newline-separated key:value pairs."""
    f = open(save_path, "w")
    for k, v in var_dict.items():
        if type(v) is pd.DataFrame:
            continue
        f.write(f"{k}: {v}\n")
    f.close()
    return


def make_learning_curve(model,
                        data_train,
                        label_train,
                        savepath,
                        train_sizes=np.linspace(0.1, 1, 10)):
    ret = sklearn.model_selection.learning_curve(model, data_train, label_train, train_sizes=train_sizes,
                                                 random_state=random_state)
    plt.plot(ret[0], ret[2], ".-")
    plt.xlabel("Data samples")
    plt.ylabel("Score")
    plt.title(f"Learning curve: {label_train.name}")
    plt.savefig(savepath)
    plt.close()
    return


def get_permutation_scores(model, data_train, label_train, savepath, n_permutations=200) -> (float, float):
    score, perm_scores, pvalue = sklearn.model_selection.permutation_test_score(model, data_train, label_train,
                                                                                n_permutations=200, random_state=random_state)
    f = open(savepath, "w")
    f.write(f"Score: {score}\n")
    f.write(f"p value: {pvalue}")
    f.close()
    return score, pvalue


def plot_shap(model, data_train, target_name, savepath, num_kmeans=10):
    data_train_summary = shap.kmeans(data_train, num_kmeans)
    explainer = shap.KernelExplainer(model.predict, data_train_summary)

    shap_values = explainer.shap_values(data_train)
    shap.summary_plot(shap_values, data_train, title=f"Features predictive of {target_name}", show=False)
    plt.title(f"Features predictive of {target_name}")
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()
    return shap_values


def train_and_plot_performance(train: pd.DataFrame,
                               valid: pd.DataFrame,
                               target_features: list,
                               predicting_features: list,
                               model: RandomForestRegressor,
                               model_params: dict = None,
                               save_dir: str = None,
                               save_prefix: str = None):
    """
    Creates plots with correlations for each target feature based.
    Parameters
    ----------
    train : pd.DataFrame
        Training set
    valid : pd.DataFrame
        Validation set
    target_features : list
        List of features to predict
    predicting_features : list
        List of features to use for predictions
    model
        sklearn model to use for predictions
    model_params : dict
        Dictionary for model parameters
    save_dir : str
        Path for where to save figures.
    save_prefix : str
        Prefix to use for the filename

    Returns
    -------
    scores
        Model scores keyed by target
    models
        Models keyed by target
    predicted_vals
        Predicted values for the validation set, keyed by target.
    true_vals
        True values for the validation set, keyed by target.
    """
    if model_params is None:
        model_params = {}
    if isinstance(predicting_features, str):
        predicting_features = [predicting_features]
    scores = {}
    models = {}
    true_vals = {}
    predicted_vals = {}
    for target in target_features:
        m = model(**model_params)
        retain_obs = ~(train[target].isna())
        m.fit(train.loc[retain_obs, predicting_features], train.loc[retain_obs, target])

        retain_obs = ~(valid[target].isna())
        score = m.score(valid.loc[retain_obs, predicting_features], valid.loc[retain_obs, target])
        val_feat = valid.loc[retain_obs, predicting_features]
        val_target = valid.loc[retain_obs, target]
        val_pred = m.predict(val_feat)

        plt.figure()
        plt.plot(val_target, val_pred, ".", label="Prediction")
        plt.plot([np.min(val_target), np.max(val_target)], [np.min(val_target), np.max(val_target)],
                 label="True relation")
        plt.xlabel("True value")
        plt.ylabel("Predicted value")
        plt.annotate(f"R^2 = {round(score, 3)}", (1, 0), xycoords="axes fraction", ha="right", va="bottom")
        r, p = pearsonr(val_pred, val_target)
        n = len(val_target)
        plt.annotate(f"r = {round(r, 3)}", (1, 0.225), xycoords="axes fraction", ha="right", va="bottom")
        plt.annotate(f"p = {'{:,.3E}'.format(p, 3)}", (1, 0.15), xycoords="axes fraction", ha="right", va="bottom")
        plt.annotate(f"n = {round(n, 3)}", (1, 0.75), xycoords="axes fraction", ha="right", va="bottom")
        plt.legend()
        models[target] = m
        scores[target] = score
        predicted_vals[target] = val_pred
        true_vals[target] = val_target
        if save_dir is not None:
            plt.savefig(f"{save_dir}/{save_prefix}_predicting_{target}")
            plt.close()
    return scores, models, predicted_vals, true_vals


def cv_eval(model,
            model_params: dict,
            train_data: pd.DataFrame,
            predicting_features: List[str],
            target: str):
    """
    Perform 5-repeated 2-fold cross-validation on the specified model and features/targets of the data.
    Parameters
    ----------
    model
        Function for instantiating the model; e.g., model = sklearn.ensemble.RandomForestRegressor  (note this is the
        function, not the instantiated model itself).
    model_params : dict
        Parameters to pass to the model instantiation function.
    train_data : pd.DataFrame
        DataFrame containing the data to train the model.
    predicting_features : List[str]
        Features in `train_data` to use for predicting the target.
    target : str
        Target to predict using the `predicting_features`.

    Returns
    -------
    list
        List of scores for the different folds.
    """
    scores = []
    for i in range(5):
        kf = KFold(n_splits=2, shuffle=True, random_state=i)
        train_data_copy = train_data[predicting_features + [target]].dropna(axis=0)
        splits = kf.split(train_data_copy)
        for train_idx, valid_idx in splits:
            train = train_data_copy.iloc[train_idx, :].copy()
            valid = train_data_copy.iloc[valid_idx, :].copy()
            r = model(**model_params)
            r.fit(train.loc[:, predicting_features], train.loc[:, target])
            score = r.score(valid[predicting_features], valid[target])
            scores.append(score)
    return scores


def cv_eval2(model,
             model_params: dict,
             train_data: pd.DataFrame,
             predicting_features: List[str],
             target: str):
    """
    Perform 5-repeated 2-fold cross-validation on the specified model and features/targets of the data.
    Parameters
    ----------
    model
        Function for instantiating the model; e.g., model = sklearn.ensemble.RandomForestRegressor  (note this is the
        function, not the instantiated model itself).
    model_params : dict
        Parameters to pass to the model instantiation function.
    train_data : pd.DataFrame
        DataFrame containing the data to train the model.
    predicting_features : List[str]
        Features in `train_data` to use for predicting the target.
    target : str
        Target to predict using the `predicting_features`.

    Returns
    -------
    list
        List of scores for the different folds.
    """

    scores = []
    bins = 4
    bins = np.linspace(np.min(train_data.loc[:, [target]]), np.max(train_data.loc[:, [target]]), bins+2)

    for i in range(5):
        kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=i)
        to_keep = ~train_data[target].isna()
        train_data_copy = train_data[predicting_features + [target]][to_keep]
        bin_id = np.digitize(train_data_copy.loc[:, [target]], bins[1:-1])
        splits = kf.split(train_data_copy, bin_id)
        for train_idx, valid_idx in splits:
            train = train_data_copy.iloc[train_idx, :].copy()
            valid = train_data_copy.iloc[valid_idx, :].copy()
            r = model(**model_params)
            r.fit(train.loc[:, predicting_features], train.loc[:, target])
            score = r.score(valid[predicting_features], valid[target])
            scores.append(score)
    return scores

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer, label_binarize


def cv_eval3(model,
             model_params: dict,
             train_data: pd.DataFrame,
             predicting_features: List[str],
             target: str,
             only_all=False):
    """
    Perform 5-repeated 2-fold cross-validation on the specified model and features/targets of the data.
    Parameters
    ----------
    model
        Function for instantiating the model; e.g., model = sklearn.ensemble.RandomForestRegressor  (note this is the
        function, not the instantiated model itself).
    model_params : dict
        Parameters to pass to the model instantiation function.
    train_data : pd.DataFrame
        DataFrame containing the data to train the model.
    predicting_features : List[str]
        Features in `train_data` to use for predicting the target.
    target : str
        Target to predict using the `predicting_features`.
    only_all : bool
        Whether to evaluate the model only on OVR (all classes) or on each class separately.
    Returns
    -------
    list
        List of scores for the different folds.
    """
    scores = []
    aucs = {"all": []}
    possible_targets = list(train_data[target].unique())
    for p in possible_targets:
        aucs[p] = []
    for i in range(5):
        kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=i)
        to_keep = ~train_data[target].isna()
        train_data_copy = train_data[predicting_features + [target]][to_keep]
        splits = kf.split(train_data_copy, train_data_copy[target])

        for train_idx, valid_idx in splits:
            train = train_data_copy.iloc[train_idx, :].copy()
            valid = train_data_copy.iloc[valid_idx, :].copy()
            r = model(**model_params)
            r.fit(train.loc[:, predicting_features], train.loc[:, target])
            score = r.score(valid[predicting_features], valid[target])
            scores.append(score)

            y_preds = r.predict_proba(valid.loc[:, predicting_features])
            label_binarizer = LabelBinarizer().fit(train.loc[:, target])
            y_onehot_test = label_binarizer.transform(valid[target])
            auc = roc_auc_score(y_onehot_test, y_preds, multi_class='ovr')
            aucs["all"].append(auc)
            if not only_all:
                for label in possible_targets:
                    y_onehot_test = label_binarize(valid[target], classes=[label])
                    label_idx = list(r.classes_).index(label)
                    auc = roc_auc_score(y_onehot_test, y_preds[:, label_idx])
                    aucs[label].append(auc)

    return scores, aucs

from collections import defaultdict as dd
from sklearn import metrics
def cv_eval4(model, model_params, train_data, predicting_features, target, remove_input_nan=False):
    """Does 5x2 evaluation of the model using the specified features/targets of the data."""
    scores = []
    aucs = {"all": []}
    roc_curves = {"all": []}

    possible_targets = list(train_data[target].unique())
    for p in possible_targets:
        aucs[p] = []
        roc_curves[p] = []
    rep_idx = -1
    for i in range(5):
        kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=i)
        to_keep = ~train_data[target].isna()
        train_data_copy = train_data[predicting_features + [target]][to_keep]
        splits = kf.split(train_data_copy, train_data_copy[target])

        for train_idx, valid_idx in splits:
            rep_idx += 1
            train = train_data_copy.iloc[train_idx, :].copy()
            valid = train_data_copy.iloc[valid_idx, :].copy()
            r = model(**model_params)
            r.fit(train.loc[:, predicting_features], train.loc[:, target])
            score = r.score(valid[predicting_features], valid[target])
            scores.append(score)

            y_preds = r.predict_proba(valid.loc[:, predicting_features])
            label_binarizer = LabelBinarizer().fit(train.loc[:, target])
            y_onehot_test = label_binarizer.transform(valid[target])
            auc = roc_auc_score(y_onehot_test, y_preds, multi_class='ovr')
            aucs["all"].append(auc)

            for label in possible_targets:
                y_onehot_test = label_binarize(valid[target], classes=[label])
                label_idx = list(r.classes_).index(label)
                auc = roc_auc_score(y_onehot_test, y_preds[:, label_idx])
                aucs[label].append(auc)
                fpr, tpr, thresh = roc_curve(y_onehot_test, y_preds[:, label_idx])

            # label_idx = list(r.classes_).index(label)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for ii in range(3):
                fpr[ii], tpr[ii], _ = metrics.roc_curve(y_onehot_test, y_preds[:, ii])
                roc_auc[ii] = metrics.auc(fpr[ii], tpr[ii])

            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[j] for j in range(3)]))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for ii in range(3):
                mean_tpr += np.interp(all_fpr, fpr[ii], tpr[ii])

            # Finally average it and compute AUC
            mean_tpr /= 3
            roc_curves[rep_idx] = [all_fpr, mean_tpr]
    return scores, aucs, roc_curves


from sklearn.model_selection import RepeatedStratifiedKFold
def cv_eval5(model, model_params, train_data, predicting_features, target, remove_input_nan=False):
    scores = []
    bins = 4
    # bin_width = (np.max(train_data.loc[:, [target]]) - np.min(train_data.loc[:, [target]])) / bins
    # bins = np.linspace(np.min(train_data.loc[:, [target]]), np.max(train_data.loc[:, [target]]), bins+2)
    # bin_id = train_data.loc[:, [target]] // bin_width
    aucs = {"all": []}
    roc_curves = {"all": []}

    possible_targets = list(train_data[target].unique())
    for p in possible_targets:
        aucs[p] = []
        roc_curves[p] = []
    rep_idx = -1
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=0)

    to_keep = ~train_data[target].isna()
    train_data_copy = train_data[predicting_features + [target]][to_keep]
    # bin_id = np.digitize(train_data_copy.loc[:, [target]], bins[1:-1])
    # bin_id = train_data_copy.loc[:, [target]] // bin_width

    # splits = kf.split(train_data_copy, )
    rep_idx
    for train_idx, valid_idx in cv.split(train_data_copy, train_data_copy[target]):
        rep_idx += 1
        # print(valid_idx.shape)
        train = train_data_copy.iloc[train_idx, :].copy()
        valid = train_data_copy.iloc[valid_idx, :].copy()
        r = model(**model_params)
        r.fit(train.loc[:, predicting_features], train.loc[:, target])
        score = r.score(valid[predicting_features], valid[target])
        scores.append(score)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        y_score = r.predict_proba(valid.loc[:, predicting_features])
        n_classes = y_score.shape[1]
        # labels =
        for i, p_label in enumerate(possible_targets):
            fpr[i], tpr[i], _ = metrics.roc_curve((valid[target] == p_label), y_score[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
            interp_tpr = np.interp(mean_fpr, fpr[i], tpr[i])
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc[i])
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
    return scores, mean_auc, mean_fpr, mean_tpr

from matplotlib import pyplot as plt
from sklearn.dummy import DummyClassifier
def cv_eval6(model,
             model_params: dict,
             train_data: pd.DataFrame,
             predicting_features: List[str],
             target: str,
             n_clfs: int = 100):
    """
    Perform 5-repeated 2-fold cross-validation on the specified model and features/targets of the data.
    Parameters
    ----------
    model
        Function for instantiating the model; e.g., model = sklearn.ensemble.RandomForestRegressor  (note this is the
        function, not the instantiated model itself).
    model_params : dict
        Parameters to pass to the model instantiation function.
    train_data : pd.DataFrame
        DataFrame containing the data to train the model.
    predicting_features : List[str]
        Features in `train_data` to use for predicting the target.
    target : str
        Target to predict using the `predicting_features`.
    n_clfs : int
        Number of dummy classifiers to use for computing p-values for the AUC distribution.
    Returns
    -------
    list
        List of scores for the different folds.
    """
    X = train_data[predicting_features]
    y = train_data[target]
    classifier = model(**model_params)
    # Set up auto repeated stratified cross-validation (5 repetitions of 2-fold cross-validation)
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=0)

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    pvals = []
    dummy_auc_dist = []
    possible_targets = list(train_data[target].unique())
    all_auc = dd(list)
    for train, test in cv.split(X, y):
        # train model
        classifier.fit(X.iloc[train], y.iloc[train])
        y_score = classifier.predict_proba(X.iloc[test])

        # roc for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i, label in enumerate(possible_targets):
            label_idx = list(classifier.classes_).index(label)
            fpr[i], tpr[i], _ = metrics.roc_curve((y.iloc[test] == label), y_score[:, label_idx])

            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
            loc_auc = metrics.roc_auc_score(y.iloc[test], y_score, multi_class="ovr")
            all_auc[label].append(loc_auc)
            # interpolate ROC curve
            interp_tpr = np.interp(mean_fpr, fpr[i], tpr[i])
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc[i])

            dummy_aucs = np.zeros(n_clfs)
            for j in range(n_clfs):
                dummy_clf = DummyClassifier(strategy='uniform', random_state=j)
                dummy_clf.fit(X.iloc[train], y.iloc[train])

                dummy_predictions = dummy_clf.predict_proba(X.iloc[test])
                dummy_aucs[j] = roc_auc_score(y.iloc[test], dummy_predictions, multi_class="ovr")
            dummy_auc_dist.append(dummy_aucs)
            p_value = (np.sum(dummy_aucs >= metrics.auc(fpr[i], tpr[i])) + 1) / (n_clfs + 1)
            pvals.append(p_value)

    # Compute mean and std for AUC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)

    std_auc = np.std(aucs)
    return pvals, mean_auc, mean_fpr, mean_tpr, all_auc


def cv_eval_stratified(model, model_params, train_data, predicting_features, target, remove_input_nan=False):
    """Does 5x2 evaluation of the model using the specified features/targets of the data."""
    scores = []
    bins = 5
    bin_width = (np.max(train_data.loc[:, [target]]) - np.min(train_data.loc[:, [target]]))/bins
    bin_id = train_data.loc[:, [target]] // bin_width
    # bin_id.fillna(0, inplace=True)
    print(bin_id.shape)
    bin_id.dropna(inplace=True)
    for i in range(5):
        kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=i)
        train_data_copy = train_data[predicting_features + [target]].dropna(axis=0)
        print(train_data_copy.shape)
        # splits = kf.split(train_data_copy)
        splits = kf.split(train_data_copy, bin_id)
        for train_idx, valid_idx in splits:
            train = train_data_copy.iloc[train_idx, :].copy()
            valid = train_data_copy.iloc[valid_idx, :].copy()
            r = model(**model_params)
            r.fit(train.loc[:, predicting_features], train.loc[:, target])
            score = r.score(valid[predicting_features], valid[target])
            scores.append(score)
    return scores


def main(data_path,
         features_to_predict_target: list,
         target_features: str,
         required_features: list = None,
         features_to_onehot_encode: list = None,
         features_to_drop: list = None,
         stratification_features: list = None,
         train_size: float = 0.7,
         save_path: str = None):
    if save_path is not None:
        save_path = make_directory_unique(save_path)
        # Write out summary
        write_project_summary(vars(), os.path.join(save_path, "summary.txt"))
    data, stratification_features = load_data(data_path,
                                              required_features=required_features + [target_features],
                                              features_to_onehot_encode=features_to_onehot_encode,
                                              features_to_drop=features_to_drop,
                                              stratification_features=stratification_features)
    data_train, data_val = train_test_split(data,
                                            train_size=train_size,
                                            stratify=data[stratification_features].fillna(0),
                                            random_state=random_state)
    train_subjects = set(data_train.index)
    val_subjects = set(data_val.index)


if __name__ == "__main__":
    parser = ArgumentParser(description="Analyze retinal & AD dataset.")
    parser.add_argument("data_path", type=str, help="Path to the data file.")
    parser.add_argument("save_path", type=str, help="Path where to store outputs. If it exists, name will be mangled to prevent overwrite.")

    args = parser.parse_args()
    main(args.data_path, save_path=args.save_path)
