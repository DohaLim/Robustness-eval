import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import logging
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

THRESHOLDS = {
    'adult': 0.32,
    'bank': 0.3,
    'blood': 0.21,
    'credit-g': 0.21,
    'diabetes': 0.23,
    'heart': 0.39,
    'cultivar': 0.17,
    'myocardial': 0.15
}


def get_pool_test(data):
    file_name = f"./data/{data}.csv"
    df = pd.read_csv(file_name)
    target_variable = df.columns[-1]
    categorical_indicator = [
        True if (dt == np.dtype('O') or pd.api.types.is_string_dtype(dt)) else False
        for dt in df.dtypes.tolist()
    ][:-1]
    X = df.convert_dtypes()
    y = df[target_variable]
    label_list = np.unique(y).tolist()
    X_pool, X_test, y_pool, y_test = train_test_split(
        X.drop(columns=[target_variable]),
        y,
        test_size=0.2,
        random_state=0,
        stratify=y
    )
    return df, X_pool, X_test, y_pool, y_test, target_variable, label_list, categorical_indicator

def list_correlation(data_corr, target_attr, label_list):
    for label in label_list:
        print(f"\nCorrelation with class '{label}':")
        print(data_corr[label][abs(data_corr[label]) >= 0.1])
    data_corr[target_attr] = data_corr[target_attr].round(2)
    data_corr[abs(data_corr[target_attr]) >= 0.2] \
    .sort_values(by=target_attr, key=lambda x: abs(x), ascending=False)[[target_attr]]

def get_elbow_thresholds(data_corr, target_attr, top_k=10, min_corr=0.2):
    corr_series = data_corr[target_attr].drop(index=target_attr)
    corr_abs = corr_series.abs()
    sorted_corr = corr_abs[corr_abs >= min_corr].sort_values(ascending=False)

    if len(sorted_corr) < 3:
        return None, None

    diffs = sorted_corr.values[:-1] - sorted_corr.values[1:]
    idx_sorted = sorted_corr.index

    top_indices = diffs.argsort()[::-1][:2]
    thresholds = sorted([sorted_corr.values[i+1] for i in top_indices], reverse=True)
    threshold1 = thresholds[0]
    threshold2 = thresholds[1] if len(thresholds) > 1 else None
    logging.info(f"Detected threshold1: {threshold1:.2f}, threshold2: {threshold2:.2f}")
    return threshold1, threshold2

def plot_elbow_method(data_corr, target_attr, threshold1, threshold2, min_corr=0.2):
    sorted_data = data_corr[abs(data_corr[target_attr]) >= min_corr].sort_values(by=target_attr, key=lambda x: abs(x), ascending=False)[[target_attr]]
    sorted_data[target_attr] = sorted_data[target_attr].abs()
    sorted_data = sorted_data.drop(index=target_attr)

    attributes = sorted_data.index
    values = sorted_data[target_attr].values

    plt.figure(figsize=(12, 6))
    plt.plot(attributes, values, marker='o', linestyle='-', color='b')
    plt.ylim(min_corr, 0.6)
    plt.axhline(y=threshold1, color='red', linestyle='--', linewidth=0.8, label=f'Threshold > {threshold1}')
    plt.axhline(y=threshold2, color='red', linestyle='--', linewidth=0.8, label=f'Threshold > {threshold2}')
    plt.title(f'{target_attr} Correlation Elbow Plot (Absolute Values)', fontsize=16)
    plt.xlabel('Attributes', fontsize=12)
    plt.ylabel('Absolute Correlation Value', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('visualization/Correlation Elbow Plot')

def get_correlation(X_pool, y_pool, is_cat, target_attr, label_list):
    X_pool = X_pool.convert_dtypes()
    feature_name_list = []
    selected_column = X_pool.columns.tolist()
    sel_cat_idx = [X_pool.columns.tolist().index(col_name) for col_name in selected_column]

    is_cat_sel = np.array(is_cat)[sel_cat_idx]

    for cidx, cname in enumerate(selected_column):
        if is_cat_sel[cidx] == False:
            feature_name_list.append(cname)
    data = pd.DataFrame(X_pool)
    data_numeric = data[feature_name_list]
    data_combined = data_numeric

    if len(feature_name_list) != len(selected_column):
      data_cat = data.drop(columns=feature_name_list)
      data_cat_encoded = pd.get_dummies(data_cat)
      data_cat_encoded.columns = [
            f"{col[:col.rfind('_')]}__{col[col.rfind('_') + 1:]}" if "_" in col else col
            for col in data_cat_encoded.columns
        ]
      data_combined = pd.concat([data_numeric, data_cat_encoded], axis=1)

    data_combined[target_attr] = y_pool
    data_combined[target_attr] = data_combined[target_attr].apply(lambda x: 1 if x == label_list[1] else 0)
    data_corr = data_combined.corr()
    return data_corr


def get_golden_standard(data_corr, target_attr, label_list, threshold):
    golden_features = [feature for feature in data_corr[abs(data_corr[target_attr]) >= threshold][target_attr].index if feature != target_attr]
    golden_features_nm = list(set(feature.split('__')[0] for feature in golden_features))

    target_standard_relations = {label_list[0]: {}, label_list[1]: {}}
    for feature in golden_features_nm:
        target_standard_relations[label_list[0]][feature] = []
        target_standard_relations[label_list[1]][feature] = []

    for feature in golden_features:
      correlation_value = data_corr[target_attr][feature]
      if '__' in feature: # categorical
        if correlation_value > 0:
          target_standard_relations[label_list[1]][feature.split('__')[0]].append(feature.split('__')[1])
        else:
          target_standard_relations[label_list[0]][feature.split('__')[0]].append(feature.split('__')[1])
      elif correlation_value > 0:  # positive
          target_standard_relations[label_list[0]][feature] = 'negative'
          target_standard_relations[label_list[1]][feature] = 'positive'
      elif correlation_value < 0:  # negative
          target_standard_relations[label_list[0]][feature] = 'positive'
          target_standard_relations[label_list[1]][feature] = 'negative'

    for feature in target_standard_relations[label_list[0]]:
        if not target_standard_relations[label_list[0]][feature]:
            matching_features = [f for f in data_corr.columns if f.startswith(feature + '__')]
            max_corr_feature = max(matching_features, key=lambda f: abs(data_corr[target_attr][f]) if data_corr[target_attr][f] < 0 else float('-inf'))
            if max_corr_feature:
                target_standard_relations[label_list[0]][feature].append(max_corr_feature.split('__')[1])
                golden_features.append(max_corr_feature)

    for feature in target_standard_relations[label_list[1]]:
        if not target_standard_relations[label_list[1]][feature]:
            matching_features = [f for f in data_corr.columns if f.startswith(feature + '__')]
            max_corr_feature = max(matching_features, key=lambda f: abs(data_corr[target_attr][f]) if data_corr[target_attr][f] > 0 else float('-inf'))
            if max_corr_feature:
                target_standard_relations[label_list[1]][feature].append(max_corr_feature.split('__')[1])
                golden_features.append(max_corr_feature)

    return golden_features_nm, target_standard_relations

def find_numerical_value(X_pool, y_pool, golden_feature, direction, label_list):
    values = X_pool[golden_feature].values
    y_pool_bin = np.array([label_list.index(k) for k in y_pool])

    unique_values = np.unique(values)
    thresholds = np.sort(unique_values)

    best_threshold = None
    best_auc = 0
    aucs = []

    for threshold in thresholds:
        if direction == 'positive':
          binary_feature = (values <= threshold).astype(int)
        else:
          binary_feature = (values >= threshold).astype(int)
        auc = roc_auc_score(y_pool_bin, binary_feature)
        aucs.append(auc)

        if auc > best_auc:
            best_auc = auc
            best_threshold = threshold

    min_value = np.min(unique_values)
    max_value = np.max(unique_values)
    min_max_values = (min_value, max_value)

    return best_threshold, thresholds, aucs, min_max_values

def plot_thresholds_aucs(golden_feature, threshold_list, auc_list):
    plt.figure(figsize=(12, 8))
    plt.plot(threshold_list, auc_list, label=f"{golden_feature} AUC")
    plt.xlabel("Threshold")
    plt.ylabel("AUC")
    plt.title(f"AUC by Threshold for {golden_feature}")
    plt.legend()
    plt.grid(True)
    plt.show()

def get_golden_values(golden_relations, X_pool, y_pool, label_list, plot=False):
    golden_values = {label: {} for label in label_list}
    min_max_values = {label: {} for label in label_list}
    done_variables = {}

    for label in label_list:
        relations = golden_relations[label]
        for var, val in relations.items():
            if isinstance(val, list):
                golden_values[label][var] = val
            else:
                if var not in done_variables:
                    optimal_threshold, thresholds, aucs, feature_min_max = find_numerical_value(
                        X_pool, y_pool, var, val, label_list
                    )
                    if plot:
                        plot_thresholds_aucs(var, thresholds, aucs)
                    done_variables[var] = (optimal_threshold, feature_min_max)
                optimal_threshold, feature_min_max = done_variables[var]
                golden_values[label][var] = optimal_threshold
                min_max_values[label][var] = feature_min_max

    return golden_values, min_max_values

def calculate_distance(row, target_values, feature_min_max, target_relations, label):
    distance = 0
    for var, target_value in target_values.items():
        if var in row.index:
            # Categorical
            if isinstance(target_value, list):
                if row[var] in target_value:
                    distance += 0
                elif any(row[var] in vals for lbl, vals in target_relations.items() if lbl != label):
                    distance += 1
                else:
                    distance += 0.5
            # Numerical
            else:
                min_val, max_val = feature_min_max[var]
                normalized_value = (row[var] - min_val) / (max_val - min_val + 1e-8)
                normalized_target = (target_value - min_val) / (max_val - min_val + 1e-8)
                penalty_negative = abs((min_val - min_val) / (max_val - min_val + 1e-8) - normalized_target)
                penalty_positive = abs((max_val - min_val) / (max_val - min_val + 1e-8) - normalized_target)
                if target_relations[label][var] == 'negative':
                    # best
                    if normalized_value < normalized_target:
                        distance += abs(normalized_value - normalized_target)
                    # worst
                    else:
                        distance += abs(normalized_value - normalized_target) + penalty_negative
                elif target_relations[label][var] == 'positive':
                    # best
                    if normalized_value > normalized_target:
                        distance += abs(normalized_value - normalized_target)
                    # worst
                    else:
                        distance += abs(normalized_value - normalized_target) + penalty_positive
    return distance


def setup(data):
    df, X_pool, X_test, y_pool, y_test, target_variable, label_list, is_cat = get_pool_test(data)
    X_all = df.drop(target_variable, axis=1)
    data_corr = get_correlation(X_pool, y_pool, is_cat, target_variable, label_list)
    data_corr[target_variable] = data_corr[target_variable].round(2)
    threshold = THRESHOLDS[data] if data in THRESHOLDS else None
    if not threshold:
        list_correlation(data_corr, target_variable, label_list)
        threshold1, threshold2 = get_elbow_thresholds(data_corr, target_variable)
        plot_elbow_method(data_corr, target_variable, threshold1, threshold2)
        return f"Decide the threshold for data {data}"
    else:
        golden_variables, golden_relations = get_golden_standard(data_corr, target_variable, label_list, threshold)
        golden_values, min_max_values = get_golden_values(golden_relations, X_pool, y_pool, label_list)

        setup_info = {
            "data": data,
            "label_list": label_list,
            "target_variable": target_variable,
            "is_cat": is_cat,
            "golden_variables": golden_variables,
            "golden_relations": golden_relations,
            "golden_values": golden_values,
            "min_max_values": min_max_values,
            "X_all": X_all,
            "X_pool": X_pool,
            "y_pool": y_pool,
            "X_test": X_test,
            "y_test": y_test
        }
        return setup_info

def get_train(setup_info, shot, sampling_method, seed):
    logging.info(f"Sampling {shot} examples with method {sampling_method}")
    pool = setup_info["X_pool"].copy()
    pool[setup_info["target_variable"]] = setup_info["y_pool"]

    if sampling_method not in ["random", "best", "worst"]:
        raise ValueError("Unknown method. Choose from ['random', 'best', 'worst'].")

    sampled_list = []
    shot_per_class = shot // len(setup_info["label_list"])
    remainder = shot % len(setup_info["label_list"])

    for idx, label in enumerate(setup_info["label_list"]):
        class_group = pool[pool[setup_info["target_variable"]] == label].copy()
        sample_num = shot_per_class + (1 if idx < remainder else 0)

        if sampling_method == "random":
            sampled = class_group.sample(n=sample_num, random_state=seed)

        elif sampling_method in ["best", "worst"] and setup_info["golden_values"] and setup_info["golden_relations"]:
            if len(setup_info["golden_values"][label])==0:
              sampled = class_group.sample(n=sample_num, random_state=seed)
            else:
              class_group["distance_to_standard"] = class_group.apply(
                  lambda row: calculate_distance(
                      row,
                      setup_info["golden_values"][label],
                      setup_info["min_max_values"][label],
                      setup_info["golden_relations"],
                      label
                  ), axis=1
              )
              if sampling_method == "best":
                  probabilities = 1 / (class_group["distance_to_standard"] + 1e-8)
              else:  # worst
                  probabilities = class_group["distance_to_standard"]
              probabilities /= probabilities.sum()
              sampled = class_group.sample(n=sample_num, weights=probabilities, random_state=seed)
        sampled_list.append(sampled)
    train = pd.concat(sampled_list)
    y_train = train[setup_info["target_variable"]].to_numpy()
    X_train = train.drop(columns=[setup_info["target_variable"], "distance_to_standard"], errors="ignore")
    return X_train, y_train

def swap_golden_variable_values(df, variable, target_attr, label_list):
    """
    if the number of samples are balanced
    """
    variable_values = [df[df[target_attr] == label][variable].values for label in label_list]
    shifted_values = variable_values[-1:] + variable_values[:-1]
    for label, new_values in zip(label_list, shifted_values):
        df.loc[df[target_attr] == label, variable] = new_values
    return df

def swap_golden_variable_values_unbalanced(df, variable, target_attr, label_list, seed=0):
    """
    if the number of samples are unbalanced
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    values_by_label = {lab: out.loc[out[target_attr] == lab, variable].values for lab in label_list}
    shifted = [values_by_label[label_list[i - 1]] for i in range(len(label_list))]

    for lab, new_vals in zip(label_list, shifted):
        idx = out[out[target_attr] == lab].index
        need = len(idx)
        cur = len(new_vals)
        if cur == 0:
            continue
        if cur < need:
            reps = int(np.ceil(need / cur))
            filled = np.tile(new_vals, reps)[:need]
        else:
            perm = rng.permutation(cur)
            filled = new_vals[perm][:need]
        out.loc[idx, variable] = filled
    return out

def convert_to_binary_vectors(setup_info, fct_strs_all, fct_names, X_train):
    X_train_all_dict = {}
    X_test_all_dict = {}
    executable_list = []
    for i in range(len(fct_strs_all)):
        X_train_dict, X_test_dict = {}, {}
        for label in setup_info["label_list"]:
            X_train_dict[label] = {}
            X_test_dict[label] = {}

        fct_idx_dict = {}
        for idx, name in enumerate(fct_names[i]):
            for label in setup_info["label_list"]:
                label_name = '_'.join(label.split(' '))
                if label_name.lower() in name.lower():
                    fct_idx_dict[label] = idx

        if len(fct_idx_dict) != len(setup_info["label_list"]):
            continue

        try:
            for label in setup_info["label_list"]:
                fct_idx = fct_idx_dict[label]
                exec(fct_strs_all[i][fct_idx].strip('` "'))
                X_train_each = locals()[fct_names[i][fct_idx]](X_train).astype('int').to_numpy()
                X_test_each = locals()[fct_names[i][fct_idx]](setup_info["X_test"]).astype('int').to_numpy()
                assert(X_train_each.shape[1] == X_test_each.shape[1])
                X_train_dict[label] = torch.tensor(X_train_each).float()
                X_test_dict[label] = torch.tensor(X_test_each).float()

            X_train_all_dict[i] = X_train_dict
            X_test_all_dict[i] = X_test_dict
            executable_list.append(i)
        except (ValueError, TypeError, NameError) as e: # If error occurred during the function call, remove the current trial
            logging.warning(f"Error executing function {i}: {e}")
            continue

    return executable_list, X_train_all_dict, X_test_all_dict

def evaluate(pred_probs, answers):
    result_auc = roc_auc_score(answers, pred_probs[:, 1])
    return result_auc
