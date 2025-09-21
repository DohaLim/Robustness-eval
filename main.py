#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main module for Robustness Feature Engineering Diagnosis and Evaluation
This module provides the main entry point for the system.
"""

import copy
import os
import logging
import argparse
import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

from config import (
    DEFAULT_DATA, DEFAULT_MODE, DEFAULT_LEVEL,
    DEFAULT_SHOT, DEFAULT_SAMPLING_METHOD, DEFAULT_DESC_METHOD, DEFAULT_SWAP,
    DEFAULT_SEED, DEFAULT_TEMP,
    DEFAULT_NUM_FEATURE_SET, DEFAULT_NUM_FEATURE,
    DEFAULT_LLM_ENGINE, DEFAULT_LLM_MODEL, VLLM_SERVER_URL,
    DEFAULT_OUTPUT_DIR
)

import data_utils
import llm_utils
import prompt_generator
import evaluator
from llm_utils import LLMClient


def _normalize_none(v):
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in {"none", "null", "nil", ""}:
        return None
    return v

class FeatureEvaluator:
    """
    Main class for FE Evaluation.
    Provides diagnosis and evaluation flows for LLM Feature Engineering.
    """
    def __init__(self,
                 data: str = DEFAULT_DATA,
                 mode:str = DEFAULT_MODE,
                 level: str = DEFAULT_LEVEL,
                 shot: int = DEFAULT_SHOT,
                 sampling_method: str = DEFAULT_SAMPLING_METHOD,
                 desc_method: str = DEFAULT_DESC_METHOD,
                 swap: str = DEFAULT_SWAP,
                 seed: int = DEFAULT_SEED,
                 temp: float = DEFAULT_TEMP,
                 num_feature_set: int = DEFAULT_NUM_FEATURE_SET,
                 num_feature: int = DEFAULT_NUM_FEATURE,
                 llm_engine: str = DEFAULT_LLM_ENGINE,
                 llm_name: str = DEFAULT_LLM_MODEL,
                 openai_api_key: str = None,
                 together_api_key: str = None,
                 vllm_server_url: str = None):

        self.data = data
        self.mode = mode
        self.level = level
        self.shot = shot
        self.sampling_method = sampling_method
        self.desc_method = desc_method
        self.swap = swap
        self.seed = seed
        self.temp = temp
        self.num_feature_set = num_feature_set
        self.num_feature = num_feature
        self.llm_name = llm_name
        self.setup_info = None
        self.X_train = None
        self.y_train = None
        self.llm_client = LLMClient(
            engine=llm_engine, 
            model_name=llm_name, 
            openai_api_key=openai_api_key,
            together_api_key=together_api_key,
            vllm_server_url=vllm_server_url
        )

    def prepare(self):
        self.setup_info = data_utils.setup(self.data)
        if type(self.setup_info) == str:
            logging.error(self.setup_info)
            exit()

        logging.info("Data loaded")
        logging.debug("Data: %s", self.setup_info['data'])

        logging.info("Golden Variables loaded")
        logging.debug("Golden Variables: %s", self.setup_info['golden_variables'])

        logging.info("Golden Relations loaded")
        logging.debug("Golden Relations: %s", self.setup_info['golden_relations'])

        logging.info("Golden Values loaded")
        logging.debug("Golden Values: %s", self.setup_info['golden_values'])

        if self.shot != 0:
            self.X_train, self.y_train = data_utils.get_train(
                self.setup_info, self.shot, self.sampling_method, self.seed
            )
            logging.info("Trainset prepared")
            logging.debug("X_train: %s", self.X_train)

            if self.swap is not None:
                target_attr = self.setup_info["target_variable"]
                label_list = self.setup_info["label_list"]

                if self.swap not in self.X_train.columns:
                    logging.warning("Swap feature '%s'is not in X_train.", self.swap)
                else:
                    train_df = self.X_train.copy()
                    train_df[target_attr] = self.y_train

                    counts = train_df.groupby(target_attr).size().to_list()
                    try:
                        if len(set(counts)) == 1:
                            swapped = data_utils.swap_golden_variable_values(
                                train_df, self.swap, target_attr, label_list
                            )
                            logging.info("Applied swap_golden_variable_values on '%s' (balanced classes).", self.swap)
                        else:
                            swapped = data_utils.swap_golden_variable_values_unbalanced(
                                train_df, self.swap, target_attr, label_list, seed=self.seed
                            )
                            logging.info(
                                "Applied swap_golden_variable_values_balanced on '%s' (unbalanced classes).", self.swap
                            )

                        self.X_train = swapped.drop(columns=[target_attr])
                        self.y_train = swapped[target_attr].to_numpy()
                    except Exception as e:
                        logging.exception("Error occured while Swapping %s", e)

    def run_diagnose(self):
        template = prompt_generator.generate_prompt_for_diagnosis(
            self.setup_info, self.level, self.shot, self.desc_method, self.seed,
            _class=None, X_train=self.X_train, y_train=self.y_train
        )
        logging.info("Prompt script generated")
        logging.debug("Prompt: %s", template)

        rule = f'{self.llm_name}-{self.data}-{self.level}-{self.desc_method}-{self.shot}-{self.sampling_method}-{self.swap}-{self.temp}-{self.seed}'
        logging.info("Query rule: %s", rule)

        def _diagnose_once():
            results_local = llm_utils.get_result(self.llm_name, self.data, rule, template, self.mode, self.temp, self.llm_client.engine)
            logging.info("LLM response received")
            if not results_local:
                logging.error("Empty LLM response (engine=%s). Check vLLM server/endpoint.", self.llm_client.engine)
                return None
            logging.debug("LLM response[0]: %s", results_local[0])
            parsed_local = llm_utils.parse_result(results_local, self.level, self.setup_info)
            logging.info("Parsed response ready")
            logging.debug("Parsed results: %s", parsed_local)
            score_local = evaluator.get_diagnosis_score(parsed_local, self.level, self.setup_info)
            logging.info("Diagnosis score: %s", score_local)
            return score_local

        try:
            score = _diagnose_once()
            return score
        except Exception as e:
            logging.exception("Error during diagnose pipeline: %s", e)
            # Delete the just-created cache file and retry once
            llm_dir = 'gpt/' if 'gpt' in self.llm_name.lower() else ""
            rule_file_name = f'./{"diagnose" if self.mode=="diagnose" else "evaluate"}/{self.data}/{llm_dir}{rule}.out'
            try:
                if os.path.isfile(rule_file_name):
                    os.remove(rule_file_name)
                    logging.warning("Removed corrupted output file and retrying: %s", rule_file_name)
            except Exception as rm_err:
                logging.error("Failed to remove file %s: %s", rule_file_name, rm_err)
            # retry once
            try:
                score = _diagnose_once()
                return score
            except Exception as e2:
                logging.exception("Retry failed during diagnose pipeline: %s", e2)
                return None

    def train_simple_model(self, X_train_now, label_list, shot):
        class simple_model(nn.Module):
            def __init__(self, X):
                super(simple_model, self).__init__()
                self.weights = nn.ParameterList([nn.Parameter(torch.ones(x_each.shape[1] , 1) / x_each.shape[1]) for x_each in X])

            def forward(self, x):
                x_total_score = []
                for idx, x_each in enumerate(x):
                    x_score = x_each @ torch.clamp(self.weights[idx], min=0)
                    x_total_score.append(x_score)
                x_total_score = torch.cat(x_total_score, dim=-1)
                return x_total_score

        criterion = nn.CrossEntropyLoss()
        if shot // len(label_list) == 1:
            model = simple_model(X_train_now)
            opt = Adam(model.parameters(), lr=1e-2)
            for _ in range(200):
                opt.zero_grad()
                outputs = model(X_train_now)
                preds = outputs.argmax(dim=1).numpy()
                acc = (np.array(self.y_train_num) == preds).sum() / len(preds)
                if acc == 1:
                    break
                loss = criterion(outputs, torch.tensor(self.y_train_num))
                loss.backward()
                opt.step()
        else:
            if shot // len(label_list) <= 2:
                n_splits = 2
            else:
                n_splits = 4

            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
            model_list = []
            for fold, (train_ids, valid_ids) in enumerate(kfold.split(X_train_now[0], self.y_train_num)):
                model = simple_model(X_train_now)
                opt = Adam(model.parameters(), lr=1e-2)
                X_train_now_fold = [x_train_now[train_ids] for x_train_now in X_train_now]
                X_valid_now_fold = [x_train_now[valid_ids] for x_train_now in X_train_now]
                y_train_fold = self.y_train_num[train_ids]
                y_valid_fold = self.y_train_num[valid_ids]

                max_acc = -1
                for _ in range(200):
                    opt.zero_grad()
                    outputs = model(X_train_now_fold)
                    loss = criterion(outputs, torch.tensor(y_train_fold))
                    loss.backward()
                    opt.step()
                    valid_outputs = model(X_valid_now_fold)
                    preds = valid_outputs.argmax(dim=1).numpy()
                    acc = (np.array(y_valid_fold) == preds).sum() / len(preds)
                    if max_acc < acc:
                        max_acc = acc
                        final_model = copy.deepcopy(model)
                        if max_acc >= 1:
                            break
                model_list.append(final_model)

            sdict = model_list[0].state_dict()
            for key in sdict:
                sdict[key] = torch.stack([model.state_dict()[key] for model in model_list], dim=0).mean(dim=0)

            model = simple_model(X_train_now)
            model.load_state_dict(sdict)
        return model

    def run_evaluate(self):
        templates, feature_desc = prompt_generator.generate_prompt_for_evaluation(
            self.setup_info, self.X_train, self.y_train, self.num_feature_set, self.num_feature
        )
        logging.info("Evaluation prompts generated")
        logging.debug("First prompt: %s", templates[0])

        rule = f'rule-ten-{self.data}-{self.num_feature}-{self.shot}-{self.sampling_method}-{self.seed}'
        logging.info("Evaluation rule: %s", rule)

        results = llm_utils.get_result(self.llm_name, self.data, rule, templates, self.mode, self.temp, self.llm_client.engine)
        logging.info("LLM responses received for evaluation")
        logging.debug("LLM responses[0]: %s", results[0])

        parsed_rules = llm_utils.parse_rules(results, self.setup_info["label_list"])
        logging.info("Parsed rules ready")

        n = len(parsed_rules)

        normalized_parsed_rules = llm_utils.normalize_parsed_rules(parsed_rules, self.setup_info)
        logging.info("Normalized parsed rules ready")

        scores = evaluator.get_evaluation_score(normalized_parsed_rules, self.setup_info, n)
        logging.info("Scores computed")
        logging.debug("Scores: %s", scores)

        top_3_indices, top_5_indices, bottom_3_indices = evaluator.get_top_features(scores)
        logging.info("Top features selected")
        logging.debug("Top3 indices: %s", top_3_indices)

        func_rule = f'function-ten-{self.data}-{self.num_feature}-{self.shot}-{self.sampling_method}-{self.seed}'
        logging.info("Query for function: %s", func_rule)
        fct_strs_all = llm_utils.prompt_for_function(self.data, parsed_rules, feature_desc, func_rule, self.num_feature, self.llm_client.engine)

        fct_names = []
        fct_strs_final = []
        for fct_str_pair in fct_strs_all:
            fct_pair_name = []
            if 'def' not in fct_str_pair[0]:
                continue

            for fct_str in fct_str_pair:
                fct_pair_name.append(fct_str.split('def')[1].split('(')[0].strip())
            fct_names.append(fct_pair_name)
            fct_strs_final.append(fct_str_pair)

        executable_list, X_train_all_dict, X_test_all_dict = data_utils.convert_to_binary_vectors(
            self.setup_info, fct_strs_final, fct_names, self.X_train
        )

        self.y_train_num = np.array([self.setup_info["label_list"].index(k) for k in self.y_train])
        y_test_num = np.array([self.setup_info["label_list"].index(k) for k in self.setup_info["y_test"]])

        test_outputs_all = []
        top3_outputs_all = []
        top5_outputs_all = []
        ex_bottom3_outputs_all = []
        AUCs = []

        for i in executable_list:
            X_train_now = list(X_train_all_dict[i].values())
            X_test_now = list(X_test_all_dict[i].values())

            trained_model = self.train_simple_model(X_train_now, self.setup_info["label_list"], self.shot)

            test_outputs = trained_model(X_test_now).detach().cpu()
            test_outputs = F.softmax(test_outputs, dim=1).detach()
            result_auc = data_utils.evaluate(test_outputs.numpy(), y_test_num)
            test_outputs_all.append(test_outputs)
            logging.info('AUC: %s', result_auc)
            AUCs.append(result_auc)
            if i in top_3_indices:
                top3_outputs_all.append(test_outputs)
            if i in top_5_indices:
                top5_outputs_all.append(test_outputs)
            if i in bottom_3_indices:
                ex_bottom3_outputs_all.append(test_outputs)

        # Ensemble
        test_outputs_all = np.stack(test_outputs_all, axis=0)
        top3_outputs_all = np.stack(top3_outputs_all, axis=0)
        top5_outputs_all = np.stack(top5_outputs_all, axis=0)
        ex_bottom3_outputs_all = np.stack(ex_bottom3_outputs_all, axis=0)

        ensembled_probs = test_outputs_all.mean(0)
        top3_ensembled_probs = top3_outputs_all.mean(0)
        top5_ensembled_probs = top5_outputs_all.mean(0)
        ex_bottom3_ensembled_probs = ex_bottom3_outputs_all.mean(0)

        total_auc = data_utils.evaluate(ensembled_probs, y_test_num)
        top3_total_auc = data_utils.evaluate(top3_ensembled_probs, y_test_num)
        top5_total_auc = data_utils.evaluate(top5_ensembled_probs, y_test_num)
        ex_bottom3_total_auc = data_utils.evaluate(ex_bottom3_ensembled_probs, y_test_num)
        logging.info("Ensembled AUC: %s", total_auc)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Feature Evaluation')
    parser.add_argument('--data', type=str, default=DEFAULT_DATA)
    parser.add_argument('--mode', type=str, default=DEFAULT_MODE, choices=['evaluate', 'diagnose'])
    parser.add_argument('--level', type=str, default=DEFAULT_LEVEL)
    parser.add_argument('--shot', type=int, default=DEFAULT_SHOT)
    parser.add_argument('--sampling_method', type=str, default=DEFAULT_SAMPLING_METHOD)
    parser.add_argument('--desc_method', type=str, default=DEFAULT_DESC_METHOD)
    parser.add_argument('--swap', type=str, default=DEFAULT_SWAP)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--temp', type=float, default=DEFAULT_TEMP)
    parser.add_argument('--num_feature_set', type=int, default=DEFAULT_NUM_FEATURE_SET)
    parser.add_argument('--num_feature', type=int, default=DEFAULT_NUM_FEATURE)
    parser.add_argument('--llm-engine', type=str, default=DEFAULT_LLM_ENGINE, 
                       choices=['openai', 'together', 'vllm'],
                       help='LLM engine to use')
    parser.add_argument('--llm-name', type=str, default=DEFAULT_LLM_MODEL,
                       help='Model name (e.g., gpt-3.5-turbo, meta-llama/Llama-2-7b-hf)')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--openai-api-key', type=str, help='OpenAI API key (if not in environment variables)')
    parser.add_argument('--together-api-key', type=str, help='Together AI API key (if not in environment variables)')
    parser.add_argument('--vllm-server-url', type=str, default=VLLM_SERVER_URL,
                    help='vLLM server URL (default: http://localhost:8000)')

    args = parser.parse_args()
    args.swap = _normalize_none(args.swap)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure logging to write to output directory
    log_file = os.path.join(args.output_dir, "fe_evaluation.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )

    # Initialize and run evaluator
    evaluator = FeatureEvaluator(
        data=args.data,
        mode=args.mode,
        level=args.level,
        shot=args.shot,
        sampling_method=args.sampling_method,
        desc_method=args.desc_method,
        swap=args.swap,
        seed=args.seed,
        temp=args.temp,
        num_feature_set=args.num_feature_set,
        num_feature=args.num_feature,
        llm_engine=args.llm_engine,
        llm_name=args.llm_name,
        openai_api_key=args.openai_api_key,
        together_api_key=args.together_api_key,
        vllm_server_url=args.vllm_server_url
    )

    evaluator.prepare()

    if args.mode == 'diagnose':
        evaluator.run_diagnose()
    else:
        evaluator.run_evaluate()

if __name__ == "__main__":
    main()