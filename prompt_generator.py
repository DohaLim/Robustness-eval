import os
import json
import random
import logging
import numpy as np
from pathlib import Path

TASK_DICT = {
    'adult': "Does this person earn more than 50000 dollars per year? Yes or no?",
    'bank': "Does this client subscribe to a term deposit? Yes or no?",
    'blood': "Did the person donate blood? Yes or no?",
    'credit-g': "Does this person receive a credit? Yes or no?",
    'diabetes': "Does this patient have diabetes? Yes or no?",
    'heart': "Does the coronary angiography of this patient show a heart disease? Yes or no?",
    'cultivar': 'How high will the grain yield of this soybean cultivar. Low or high?',
    'myocardial': "Does the myocardial infarction complications data of this patient show chronic heart failure? Yes or no?"
}

def serialize(row, features=None):
    target_str = f""
    included = features if features else list(row.index)

    for attr_idx, attr_name in enumerate(included):
        if attr_idx < len(list(row.index)) - 1:
            target_str += " is ".join([attr_name, str(row[attr_name]).strip(" .'").strip('"').strip()])
            target_str += ". "
        else:
            if len(attr_name.strip()) < 2:
                continue
            target_str += " is ".join([attr_name, str(row[attr_name]).strip(" .'").strip('"').strip()])
            target_str += "."
    return target_str

def fill_in_templates(fill_in_dict, template_str):
    for key, value in fill_in_dict.items():
        if key in template_str:
            template_str = template_str.replace(key, value)
    return template_str

def get_task_desc(data_name):
  task_desc = f"{TASK_DICT[data_name]}\n"
  return task_desc

def get_variable_desc(selected_column, meta_data, df_all, is_cat, seed=None, desc_method='detail'):
    if seed is not None:
        random.seed(seed)
        random.shuffle(selected_column)

    variable_name_list = []
    column_indices = [df_all.columns.tolist().index(col_name) for col_name in selected_column]
    is_cat_selected = np.array(is_cat)[column_indices]

    for column_index, column_name in enumerate(selected_column):
        description = meta_data.get(column_name, "")
        if is_cat_selected[column_index]:
            unique_values = df_all[column_name].unique().tolist()
            if len(unique_values) > 20:
                values_str = f"{unique_values[0]}, {unique_values[1]}, ..., {unique_values[-1]}"
            else:
                values_str = ", ".join(map(str, unique_values))

            if desc_method == 'detail':
                variable_name_list.append(
                    f"- {column_name}: {description} (categorical variable with categories [{values_str}])"
                )
            elif desc_method == 'simple':
                variable_name_list.append(f"- {column_name}")
            elif desc_method == 'original':
                variable_name_list.append(
                    f"- {column_name}: {description}"
                )

        else:
            if desc_method == 'detail':
                variable_name_list.append(
                    f"- {column_name}: {description} (numerical variable)"
                )
            elif desc_method == 'simple':
                variable_name_list.append(f"- {column_name}")
            elif desc_method == 'original':
                variable_name_list.append(
                    f"- {column_name}: {description}"
                )
    variable_desc = "\n".join(variable_name_list)
    return variable_desc

def get_example_desc(df_x, df_y, target_variable, selected_columns=None):
    df_incontext = df_x.copy()
    df_incontext[target_variable] = df_y
    in_context_desc = ""

    if selected_columns is not None:
        columns_to_use = selected_columns + [target_variable]
        df_incontext = df_incontext[columns_to_use]

    df_current = df_incontext.copy()
    df_current = df_current.groupby(
        target_variable, group_keys=False
    ).apply(lambda x: x.sample(frac=1))

    for icl_idx, icl_row in df_current.iterrows():
        answer = icl_row[target_variable]
        icl_row = icl_row.drop(labels=target_variable)
        in_context_desc += serialize(icl_row)
        in_context_desc += f"\nAnswer: {answer}\n"
    return in_context_desc

def get_prompt_for_rank(
    setup_info,
    shot,
    desc_method,
    seed,
    _class=None,
    df_x=None,
    df_y=None,
    template_dir="templates/",
    data_dir="data/"
):
    is_fewshot = shot > 0
    fname = "ask_importance_rank"
    if is_fewshot:
        fname += "_example"
    data = setup_info["data"]
    file_name = Path(template_dir) / f"{fname}.txt"
    meta_file_name = Path(data_dir) / f"{data}-metadata.json"

    # Load base prompt
    with open(file_name, "r") as f:
        prompt_template = f.read()

    # Load meta info
    try:
        with open(meta_file_name, "r") as f:
            meta_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.warning(f"Could not load metadata: {e}")
        meta_data = {}

    task_desc = get_task_desc(setup_info["data"])
    variable_desc = get_variable_desc(
        setup_info["X_all"].columns.tolist(),
        meta_data,
        setup_info["X_all"],
        setup_info["is_cat"],
        seed,
        desc_method
    )
    example_desc = ""
    if is_fewshot and df_x is not None and df_y is not None:
        example_desc = get_example_desc(df_x, df_y, setup_info["target_variable"])
    rank_format = "Rank:\n 1. FeatA\n 2. FeatB...."
    fill_dict = {
        "[TASK]": task_desc,
        "[FEATURES]": variable_desc,
        "[EXAMPLES]": example_desc,
        "[CLASS_NAME]": str(_class),
        "[RANK FORMAT]": rank_format,
    }
    return fill_in_templates(fill_dict, prompt_template)

def get_prompt_for_relation(
    setup_info,
    shot,
    _class=None,
    df_x=None,
    df_y=None,
    template_dir="templates/",
    data_dir="data/"
):
    is_fewshot = shot > 0
    fname = "ask_relation"
    if is_fewshot:
        fname += "_example"
    data = setup_info["data"]
    file_name = Path(template_dir) / f"{fname}.txt"
    meta_file_name = Path(data_dir) / f"{data}-metadata.json"

    # Load base prompt
    with open(file_name, "r") as f:
        prompt_template = f.read()

    # Load meta info
    try:
        with open(meta_file_name, "r") as f:
            meta_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.warning(f"Could not load metadata: {e}")
        meta_data = {}
    task_desc = get_task_desc(setup_info["data"])
    feature_format_list = []
    for label in setup_info["label_list"]:
        class_desc = f'Casual Relationship for class "{label}":'
        class_features = []
        for feature, value in setup_info["golden_relations"][label].items():
            if isinstance(value, list):
                for v in value:
                    class_features.append(f'- {feature} in {v} has a [positive/negative] correlation with class {label}')
            else:
                class_features.append(f'- {feature} has a [positive/negative] correlation with class {label}')
        class_desc += '\n' + '\n'.join(class_features)
        feature_format_list.append(class_desc)
    selected_column = sorted(set().union(*[setup_info["golden_relations"][label].keys() for label in setup_info["label_list"]]))
    feature_format_desc = '\n\n'.join(feature_format_list)
    variable_desc = get_variable_desc(selected_column, meta_data, setup_info["X_all"], setup_info["is_cat"])
    example_desc = ""
    if is_fewshot and df_x is not None and df_y is not None:
        if shot >= 16:
            example_desc = get_example_desc(df_x, df_y, setup_info["target_variable"], selected_column)
        else:
            example_desc = get_example_desc(df_x, df_y, setup_info["target_variable"])
    fill_dict = {
        "[TASK]": task_desc,
        "[CLASS_NAME]": str(_class),
        "[EXAMPLES]": example_desc,
        "[GOLDEN FEATURES]": variable_desc,
        "[FEATURE TARGET FORMAT]": feature_format_desc,
        "[FEATURE NUMBER]": str(len(selected_column))
    }
    return fill_in_templates(fill_dict, prompt_template), variable_desc

def get_prompt_for_value(
    setup_info,
    shot,
    _class=None,
    df_x=None,
    df_y=None,
    template_dir="templates/",
    data_dir="data/"
):
    is_fewshot = shot > 0

    fname = "ask_value"
    if is_fewshot:
        fname += "_example"
    data = setup_info["data"]
    file_name = Path(template_dir) / f"{fname}.txt"
    meta_file_name = Path(data_dir) / f"{data}-metadata.json"

    with open(file_name, "r") as f:
        prompt_template = f.read()
    try:
        with open(meta_file_name, "r") as f:
            meta_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.warning(f"Could not load metadata: {e}")
        meta_data = {}
    task_desc = get_task_desc(data)
    if _class:
        selected_column = [
            f for f, op in setup_info["golden_relations"][_class].items() if op is not None
        ]
    else:
        selected_column = setup_info["golden_variables"]
    variable_desc = get_variable_desc(selected_column, meta_data, setup_info["X_all"], setup_info["is_cat"])
    example_desc = ""
    if is_fewshot and df_x is not None and df_y is not None:
        if shot >= 16:
            example_desc = get_example_desc(df_x, df_y, setup_info["target_variable"], selected_column)
        else:
            example_desc = get_example_desc(df_x, df_y, setup_info["target_variable"])

    feature_condition_list = []
    for class_label, conditions in setup_info["golden_relations"].items():
        if _class and class_label != _class:
            continue
        condition_lines = [f'Condition for class "{class_label}":']
        for feature, operator in conditions.items():
            if operator is None:
                continue
            if isinstance(operator, list):
                condition_lines.append(f"- {feature} is in [Value]")
            elif operator == "negative":
                condition_lines.append(f"- {feature} is less than [Value]")
            elif operator == "positive":
                condition_lines.append(f"- {feature} is greater than [Value]")
        feature_condition_list.append("\n".join(condition_lines))
    feature_condition = "\n\n".join(feature_condition_list)
    value_format_desc = "For numerical features: [Value]\nFor categorical features: [list of Categorical Values]"
    fill_dict = {
        "[TASK]": task_desc,
        "[CLASS_NAME]": str(_class),
        "[EXAMPLES]": example_desc,
        "[FEATURES]": variable_desc,
        "[FEATURE CONDITION]": feature_condition,
        "[FEATURE NUMBER]": str(len(selected_column)),
        "[VALUE FORMAT]": value_format_desc,
    }
    return fill_in_templates(fill_dict, prompt_template), variable_desc



def generate_prompt_for_diagnosis(setup_info, level, shot=0, desc_method='detail', seed=0, _class=None, X_train=None, y_train=None):
    logging.info(f"Generating prompts for {level}")
    if level == 'lv1':
        if shot:
            template = get_prompt_for_rank(setup_info, shot, desc_method, seed, _class, X_train, y_train)
        else:
            template = get_prompt_for_rank(setup_info, shot, desc_method, seed)
    elif level == 'lv2':
        if shot:
            template = get_prompt_for_relation(setup_info, shot, _class, X_train, y_train)
        else:
            template = get_prompt_for_relation(setup_info, shot)
    else:
        if shot:
            template = get_prompt_for_value(setup_info, shot, _class, X_train, y_train)
        else:
            template = get_prompt_for_value(setup_info, shot)

    return template

def generate_prompt_for_evaluation(setup_info, X_train, y_train, num_feature_set, num_feature_rule):
    data = setup_info['data']
    logging.info(f"Generating feature rules for {data}")
    with open("templates/ask_llm.txt", "r") as f:
        prompt_type_str = f.read()
    try:
        with open(f'data/{data}-metadata.json', "r") as f:
            meta_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.warning(f"Could not load metadata: {e}")
        meta_data = {}

    target_attr = setup_info["target_variable"]
    label_list = setup_info["label_list"]
    task_desc = f"{TASK_DICT[setup_info['data']]}\n"
    df_incontext = X_train.copy()
    df_incontext[target_attr] = y_train
    feat_num = str(num_feature_rule)

    format_list = [f'{feat_num} different conditions for class "{label}":\n- [Condition]\n...' for label in label_list]
    format_desc = '\n\n'.join(format_list)

    template_list = []
    current_query_num = 0

    while True:
        if current_query_num >= num_feature_set:
            break

        # Feature bagging
        if len(df_incontext.columns) > 20:
            total_column_list = []
            for i in range(len(df_incontext.columns) // 20):
                column_list = df_incontext.columns.tolist()[:-1]
                random.shuffle(column_list)
                total_column_list.append(column_list[i*20:(i+1)*20])
        else:
            total_column_list = [df_incontext.columns.tolist()[:-1]]

        for selected_column in total_column_list:
            if current_query_num >= num_feature_set:
                break

            # Sample bagging
            threshold = 16
            if len(df_incontext) > threshold:
                sample_num = int(threshold / df_incontext[target_attr].nunique())
                df_incontext = df_incontext.groupby(
                    target_attr, group_keys=False
                ).apply(lambda x: x.sample(sample_num))

            feature_name_list = []
            sel_cat_idx = [df_incontext.columns.tolist().index(col_name) for col_name in selected_column]
            is_cat_sel = np.array(setup_info["is_cat"])[sel_cat_idx]

            for cidx, cname in enumerate(selected_column):
                if is_cat_sel[cidx] == True:
                    clist = setup_info["X_all"][cname].unique().tolist()
                    if len(clist) > 20:
                        clist_str = f"{clist[0]}, {clist[1]}, ..., {clist[-1]}"
                    else:
                        clist_str = ", ".join(clist)
                    desc = meta_data[cname] if cname in meta_data.keys() else ""
                    feature_name_list.append(f"- {cname}: {desc} (categorical variable with categories [{clist_str}])")
                else:
                    desc = meta_data[cname] if cname in meta_data.keys() else ""
                    feature_name_list.append(f"- {cname}: {desc} (numerical variable)")

            feature_desc = "\n".join(feature_name_list)

            in_context_desc = ""
            df_current = df_incontext.copy()
            df_current = df_current.groupby(
                target_attr, group_keys=False
            ).apply(lambda x: x.sample(frac=1))

            for icl_idx, icl_row in df_current.iterrows():
                answer = icl_row[target_attr]
                icl_row = icl_row.drop(labels=target_attr)
                icl_row = icl_row[selected_column]
                in_context_desc += serialize(icl_row)
                in_context_desc += f"\nAnswer: {answer}\n"

            fill_in_dict = {
                "[TASK]": task_desc,
                "[EXAMPLES]": in_context_desc,
                "{feat_num}": feat_num,
                "[FEATURES]": feature_desc,
                "[FORMAT]": format_desc
            }
            template = fill_in_templates(fill_in_dict, prompt_type_str)
            template_list.append(template)
            current_query_num += 1

    return template_list, feature_desc

def generate_prompt_for_function(parsed_rule, feature_desc, file_name):
    template_list = []
    for class_id, each_rule in parsed_rule.items():
        function_name = f'extracting_features_{class_id}'
        rule_str = '\n'.join([f'- {k}' for k in each_rule])

        fill_in_dict = {
            "[NAME]": function_name,
            "[CONDITIONS]": rule_str,
            "[FEATURES]": feature_desc
        }
        template = fill_in_templates(fill_in_dict, prompt_type_str)
        template_list.append(template)

    return template_list