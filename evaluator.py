from itertools import chain
import pandas as pd
import numpy as np
import re

######## Diagnose ########

def calculate_rankscore(ranked_features, golden_features):
    alpha = len(ranked_features) * 0.5
    ranked_indices = {feature: idx + 1 for idx, feature in enumerate(ranked_features)}
    rankscore = 0
    for feature in golden_features:
        if feature in ranked_indices:
            rank = ranked_indices[feature]
            rankscore += 1 / (2 ** ((rank - 1) / alpha))
    rankscore_max = sum(1 / (2 ** ((rank - 1) / alpha)) for rank in range(1, len(golden_features) + 1))
    normalized_rankscore = rankscore / rankscore_max if rankscore_max > 0 else 0
    return normalized_rankscore

def get_rank_score(rank_dict, golden_features):
    results = {}
    for idx in rank_dict:
        rankscore = calculate_rankscore(rank_dict[idx], golden_features)
        results[idx] = rankscore
    return results

def get_class_feature_relationship(golden_relations):
    transformed_dict = {}
    for class_key, features in golden_relations.items():
        transformed_dict[class_key] = {}
        for feature, value in features.items():
            if isinstance(value, list):  # Categorical
                for v in value:
                    transformed_dict[class_key][f'{feature} in {v}'] = 'positive'
            else:  # Numerical
                transformed_dict[class_key][feature] = value
    return transformed_dict

def get_class_relationship_score(relationship_dict, standard_operators):
    results = {}
    for idx, classes in relationship_dict.items():
        results[idx] = {}
        for class_key, features in classes.items():
            results[idx][class_key] = {}
            for feature, relationship in features.items():
                standard_operator = standard_operators[class_key].get(feature, None)
                if standard_operator is None:
                    results[idx][class_key][feature] = None
                    continue
                if relationship == standard_operator:
                    results[idx][class_key][feature] = 1
                else:
                    results[idx][class_key][feature] = 0
    return results

def get_categorical_score(standard_values, predicted_value_lists, golden_values, label, feature):
    scores = []
    if any(isinstance(sublist, list) for sublist in predicted_value_lists):
        predicted_value_lists = list(chain.from_iterable(predicted_value_lists))

    other_labels = [lbl for lbl in golden_values.keys() if lbl != label]
    other_class_values = []
    for other_label in other_labels:
        if feature in golden_values[other_label]:
            other_class_values.extend(golden_values[other_label][feature])

    matches = sum(1 for std_val in standard_values if std_val in predicted_value_lists)
    if matches > 0:
          score = matches / len(standard_values)
    else:
          if any(pred_val in other_class_values for pred_val in predicted_value_lists):
            score = 0
          else:
            score = 0.5
    scores.append(score)
    return round(sum(scores) / len(scores), 2)

def get_numerical_score(standard_value, predicted_values, min_value, max_value):
    if max_value == min_value:
        return 0.0
    scores = []
    normalized_predicted = [
            (pred - min_value) / (max_value - min_value) for pred in predicted_values
        ]
    normalized_standard = (standard_value - min_value) / (max_value - min_value)

    relative_errors = [abs(pred - normalized_standard) for pred in normalized_predicted]
    avg_error = sum(relative_errors) / len(relative_errors)
    score = max(0, 1 - avg_error)
    scores.append(score)

    return round(sum(scores) / len(scores), 2)

def get_target_value_score(golden_values, min_max_values, parsed_result, n):
    results = {}
    for idx in range(n):
        results[idx] = {}
        for label in golden_values.keys():
            results[idx][label] = {}
            for feature in golden_values[label].keys():
                standard_value = golden_values[label][feature]
                predicted_value_list = parsed_result[label][feature][idx]

                if not isinstance(predicted_value_list, list):
                    predicted_value_list = [predicted_value_list]

                if isinstance(predicted_value_list, pd.DataFrame):
                    predicted_value_list = predicted_value_list.values.flatten().tolist()

                if not predicted_value_list or all(v is None for v in predicted_value_list):
                    results[idx][label][feature] = None
                    continue

                scores = []
                for predicted_value in predicted_value_list:
                  if isinstance(standard_value, list):  # Categorical feature
                          score = get_categorical_score(
                              standard_values=standard_value,
                              predicted_value_lists=predicted_value_list,
                              golden_values=golden_values,
                              label=label,
                              feature=feature
                          )

                  else:  # Numerical feature
                          min_value, max_value = min_max_values[label][feature]
                          score = get_numerical_score(
                              standard_value=standard_value,
                              predicted_values=predicted_value_list,
                              min_value=min_value,
                              max_value=max_value
                          )
                  scores.append(score)
                  total_score = sum(scores) / len(scores)
                  results[idx][label][feature] = total_score
    return results

def get_diagnosis_score(parsed_result, level, setup_info):
    if level == 'lv1':
        score = get_rank_score(parsed_result, setup_info["golden_variables"])
        if isinstance(score, dict):
            has012 = all(k in score for k in [0, 1, 2])
            if has012:
                s1, s2, s3 = score.get(0), score.get(1), score.get(2)
                nums = [v for v in [s1, s2, s3] if isinstance(v, (int, float))]
                avg = sum(nums) / len(nums) if nums else None
                return avg

            nums = []
            for v in score.values():
                if isinstance(v, (int, float)):
                    nums.append(v)
            if nums:
                avg = sum(nums) / len(nums)
        return avg

    elif level == 'lv2':
        feature_relationship = get_class_feature_relationship(setup_info["golden_relations"])
        score = get_class_relationship_score(parsed_result, feature_relationship)

        per_feature_averages = []
        for label in setup_info["label_list"]:
            for feature in feature_relationship[label].keys():
                feature_scores = [
                    score.get(0, {}).get(label, {}).get(feature),
                    score.get(1, {}).get(label, {}).get(feature),
                    score.get(2, {}).get(label, {}).get(feature)
                ]
                valid_scores = [s if s is not None else 0 for s in feature_scores]
                if not valid_scores:
                    continue
                per_feature_averages.append(sum(valid_scores) / len(valid_scores))

        overall_avg = (sum(per_feature_averages) / len(per_feature_averages)) if per_feature_averages else 0
        return overall_avg

    else:
        score = get_target_value_score(setup_info["golden_values"], setup_info["min_max_values"], parsed_result, 3)

        per_feature_averages = []
        for label in setup_info["label_list"]:
            for feature in setup_info["golden_values"][label].keys():
                feature_scores = [
                    score.get(0, {}).get(label, {}).get(feature),
                    score.get(1, {}).get(label, {}).get(feature),
                    score.get(2, {}).get(label, {}).get(feature)
                ]
                valid_scores = [s if s is not None else 0 for s in feature_scores]
                if not valid_scores:
                    continue
                per_feature_averages.append(sum(valid_scores) / len(valid_scores))

        overall_avg = (sum(per_feature_averages) / len(per_feature_averages)) if per_feature_averages else 0
        return overall_avg

######## Evaluate ########
def extract_numerical_value(expression, feature):
    if "within range of" in expression and feature in expression:
        match = re.search(r'within range of (\[[\d,\s]+\]|\{[\d,\s]+\})', expression)
        if match:
            return match.group(1).strip()

    if "is in" in expression and feature in expression:
        parts = expression.split("is in")
        if len(parts) > 1:
            values = parts[1].strip()
            if values.startswith('[') and values.endswith(']'):
                return values

    if feature in expression:
        match = re.search(r'\[[\d,\s]+\]', expression)
        if match:
            return match.group().strip()

    for op in ['<=', '<', '>=', '>', 'is']:
        if op in expression and feature in expression:
            parts = expression.split(op)
            if len(parts) > 1:
                try:
                    value = parts[1].strip()
                    if ')' in value:
                        value = value.split(')')[0]
                    if 'else' in value:
                        value = value.split('else')[0]
                    return float(value)
                except (ValueError, TypeError) as e:
                    logging.warning(f"Error parsing numerical value from '{expression}': {e}")
                    continue
    return None

def extract_categorical_value(expression, feature):
    if "is in" in expression and feature in expression:
        parts = expression.split("is in")
        if len(parts) > 1:
            values = parts[1].strip()
            if values.startswith('[') and values.endswith(']'):
                values = values[1:-1].split(',')
                return [v.strip() for v in values]
    elif "is" in expression and feature in expression:
        parts = expression.split("is")
        if len(parts) > 1:
            values = parts[1].strip()
            if values.startswith('[') and values.endswith(']'):
                values = values[1:-1].split(',')
                return [v.strip() for v in values]
            return parts[1].strip()
    return None

def parse_results(parsed_rules, setup_info, n):
    parsed_result = {}
    for label in setup_info["label_list"]:
        parsed_result[label] = {}
        for feature in setup_info["golden_variables"]:
            parsed_result[label][feature] = [[] for _ in range(n)]
    for idx, rule_dict in enumerate(parsed_rules):
        for class_label in  setup_info["label_list"]:
            if class_label not in rule_dict:
                continue
            rules = rule_dict[class_label]
            for rule in rules:
                for feature in setup_info["golden_variables"]:
                    if feature in rule:
                        if isinstance(setup_info["golden_values"][class_label][feature], list):
                            value = extract_categorical_value(rule, feature)
                            if value is not None:
                              parsed_result[class_label][feature][idx].append(value)
                        else:
                            value = extract_numerical_value(rule, feature)
                            if value is not None:
                                if isinstance(value, str) and len(value) > 0: # within case
                                    relation = setup_info["golden_relations"][class_label][feature]
                                    str_list = list(map(int, re.findall(r'\d+', value)))
                                    if relation == 'negative':
                                        value = float(max(str_list))
                                    elif relation == 'positive':
                                        value = float(min(str_list))
                                parsed_result[class_label][feature][idx].append(value)
    return parsed_result


def get_lv1_score(rules, golden_features):
    matched_features = set()
    for rule in rules:
        for feature in golden_features:
            if re.search(rf"{feature}", rule):
                matched_features.add(feature)

    tp = len(matched_features)
    fp = len(rules) - tp
    fn = len(golden_features) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def get_lv2_score(rules, golden_relations, class_name):
    scores = []
    total_features = len(golden_relations[class_name])
    other_classes = {key: values for key, values in golden_relations.items() if key != class_name}

    for feature, condition in golden_relations[class_name].items():
        for rule in rules:
            # Numerical
            if isinstance(condition, str) and condition in ['negative', 'positive']:
                if condition == 'negative' and re.search(rf"{feature}.*<=|{feature}.*<", rule):
                    scores.append(1)
                    break
                elif condition == 'positive' and re.search(rf"{feature}.*>=|{feature}.*>", rule):
                    scores.append(1)
                    break
                elif re.search(rf"{feature} is (\d+|\[(\d+,\s*)*\d+\])", rule) or re.search(rf"{feature} is within range of", rule):
                    scores.append(0.5)
                    break

            # Categorical
            elif isinstance(condition, list):
                if feature in rule:
                    contains_other_class_value = False
                    for other_class, other_features in other_classes.items():
                        if feature in other_features:
                            other_values = other_features[feature]
                            if any(value in rule for value in other_values) and 'not' not in rule:
                                contains_other_class_value = True
                                break
                    if contains_other_class_value:
                        print('contains value from other class')
                    else:
                        scores.append(1)
                        print('no value from other class')
                    break

    return sum(scores) / len(scores) if len(scores) > 0 else 0

def calculate_average_without_none(nested_list):
    def process_sublist(sublist):
        valid_numbers = [x for x in sublist if x is not None]
        return sum(valid_numbers) / len(valid_numbers) if valid_numbers else None

    return [process_sublist(sublist) for sublist in nested_list]

def get_lv3_score(target_score_dict, label_list, golen_values):
    rule_scores = [[] for _ in range(len(target_score_dict))]

    for class_key in label_list:
        for feature_key in golen_values[class_key].keys():
            scores = [
                target_score_dict[i][class_key].get(feature_key)
                for i in range(len(target_score_dict))
            ]
            for i in range(len(rule_scores)):
                rule_scores[i].append(scores[i])

    result = calculate_average_without_none(rule_scores)
    return result

def evaluate_rules(normalized_parsed_rules, setup_info, target_score_dict):
    lv1_scores = []
    lv2_scores = []
    lv3_scores = get_lv3_score(target_score_dict, setup_info["label_list"], setup_info["golden_values"])
    cnt = 0

    for condition_set in normalized_parsed_rules:
        cnt  += 1
        lv1_rule_scores = []
        lv2_rule_scores = []

        for class_name, rules in condition_set.items():
            # Level 1
            lv1_class_score = get_lv1_score(rules, setup_info["golden_variables"])
            lv1_rule_scores.append(lv1_class_score)

            # Level 2
            lv2_class_score = get_lv2_score(rules, setup_info["golden_relations"], class_name)
            lv2_rule_scores.append(lv2_class_score)

        total_precision = sum(item['precision'] for item in lv1_rule_scores)
        total_recall = sum(item['recall'] for item in lv1_rule_scores)
        total_f1_score = sum(item['f1_score'] for item in lv1_rule_scores)

        num_scores = len(lv1_rule_scores)
        avg_precision = total_precision / num_scores
        avg_recall = total_recall / num_scores
        avg_f1_score = total_f1_score / num_scores

        avg_scores = {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1_score
        }

        lv1_scores.append(avg_scores)
        lv2_scores.append(sum(lv2_rule_scores) / len(lv2_rule_scores) if lv2_rule_scores else 0)

    return {
        'level_1_scores': lv1_scores,
        'level_2_scores': lv2_scores,
        'level_3_scores': lv3_scores
    }


def get_evaluation_score(normalized_parsed_rules, setup_info, n):
    parsed_result = parse_results(normalized_parsed_rules, setup_info, n)
    target_score_dict = get_target_value_score(setup_info["golden_values"], setup_info["min_max_values"], parsed_result, n)
    scores = evaluate_rules(normalized_parsed_rules, setup_info, target_score_dict)
    return scores

def get_top_features(scores, weights=None):
    level_1_f1_scores = [score['f1_score'] for score in scores['level_1_scores']]
    level_2_scores = [0 if x is None else x for x in scores['level_2_scores']]
    level_3_scores = [0 if x is None else x for x in scores['level_3_scores']]
    
    # e.g. weights = np.array([0.5, 0.2, 0.3])
    if weights:
        weighted_scores = np.average(
                        np.array([level_1_f1_scores, level_2_scores, level_3_scores]),
                        axis=0,
                        weights=weights
                    )
    else:
        weighted_scores = np.mean(
                    np.array([level_1_f1_scores, level_2_scores, level_3_scores]),
                    axis=0
                )
    top_3_indices = np.argsort(weighted_scores)[-3:][::-1]
    top_5_indices = np.argsort(weighted_scores)[-5:][::-1]
    bottom_3_indices = np.argsort(weighted_scores)[:3][::-1]
    return top_3_indices, top_5_indices, bottom_3_indices
