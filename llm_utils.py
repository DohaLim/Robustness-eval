import os
import re
import time
import logging
from typing import List, Optional
import requests
import openai
from tqdm import tqdm
from prompt_generator import generate_prompt_for_function

from config import (
    OPENAI_API_KEY,
    TOGETHER_API_KEY,
    DEFAULT_LLM_ENGINE,
    DEFAULT_LLM_MODEL,
)

_DIVIDER = "\n\n---DIVIDER---\n\n"
_VERSION = "\n\n---VERSION---\n\n"

API_TIMEOUT = int(os.getenv("API_TIMEOUT", "120"))
API_RETRY_ATTEMPTS = int(os.getenv("API_RETRY_ATTEMPTS", "2"))
API_RETRY_DELAY = float(os.getenv("API_RETRY_DELAY", "2.0"))


class LLMClient:
    """
    LLM client supporting OpenAI, Together AI, and vLLM
    """

    def __init__(self, engine: str = DEFAULT_LLM_ENGINE, model_name: str = DEFAULT_LLM_MODEL,
                 openai_api_key: Optional[str] = None, together_api_key: Optional[str] = None,
                 vllm_server_url: Optional[str] = None):

        self.engine = engine.lower()
        self.model_name = model_name

        # API key setup
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        self.together_api_key = together_api_key or TOGETHER_API_KEY
        self.vllm_server_url = vllm_server_url or "http://localhost:8000"

        # Validate engine
        if self.engine not in ["openai", "together", "vllm"]:
            raise ValueError(f"Unsupported LLM engine: {self.engine}. Must be 'openai', 'together', 'vllm'.")

        # Validate API keys
        if self.engine == "openai" and not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        elif self.engine == "together" and not self.together_api_key:
            raise ValueError("Together AI API key is required")

        # Initialize clients
        if self.engine == "openai":
            openai.api_key = self.openai_api_key

    def generate_text(self, prompt: str, temperature: float = 0.5, max_tokens: int = 300) -> str:
        if self.engine == "openai":
            return self._generate_text_with_openai(prompt, temperature, max_tokens)
        elif self.engine == "together":
            return self._generate_text_with_together(prompt, temperature, max_tokens)
        elif self.engine == "vllm":
            return self._generate_text_with_vllm(prompt, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")

    def _generate_text_with_vllm(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """
        Safely request to vLLM server:
        1) /v1/chat/completions (string content)
        2) /v1/chat/completions (content parts format)
        3) /v1/completions (prompt-based)
        4) /generate (native server only)
        """
        last_err = None
        base = self.vllm_server_url.rstrip("/")
        vllm_api_key = os.getenv("VLLM_API_KEY", "").strip()
        headers = {"Content-Type": "application/json"}
        if vllm_api_key:
            headers["Authorization"] = f"Bearer {vllm_api_key}"

        # 1) OpenAI chat (string content)
        chat_payload_str = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            r = requests.post(f"{base}/v1/chat/completions", headers=headers,
                            json=chat_payload_str, timeout=API_TIMEOUT)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            else:
                logging.warning(f"vLLM OpenAI-compatible {r.status_code}: {r.text}")
                if ("ChatCompletionContentPart" in r.text) or ("dict_type" in r.text):
                    raise RuntimeError("need_content_parts_retry")
        except Exception as e:
            last_err = e
            logging.debug(f"chat(string) path failed: {e}")

        # 2) OpenAI chat (content parts format)
        chat_payload_parts = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful AI assistant."}]},
                {"role": "user",   "content": [{"type": "text", "text": prompt}]},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            r = requests.post(f"{base}/v1/chat/completions", headers=headers,
                            json=chat_payload_parts, timeout=API_TIMEOUT)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            else:
                logging.warning(f"vLLM OpenAI-compatible (content parts) {r.status_code}: {r.text}")
        except Exception as e:
            last_err = e
            logging.debug(f"chat(parts) path failed: {e}")

        # 3) OpenAI completions (prompt-based)
        prompt_text = (
            "System: You are a helpful AI assistant.\n"
            f"User: {prompt}\nAssistant:"
        )
        comp_payload = {
            "model": self.model_name,
            "prompt": prompt_text,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            r = requests.post(f"{base}/v1/completions", headers=headers,
                            json=comp_payload, timeout=API_TIMEOUT)
            if r.status_code == 200:
                return r.json()["choices"][0]["text"].strip()
            else:
                logging.warning(f"vLLM OpenAI completions {r.status_code}: {r.text}")
        except Exception as e:
            last_err = e
            logging.debug(f"completions path failed: {e}")

        # 4) Native /generate (not available in OpenAI-compatible servers)
        native_payload = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
        try:
            r = requests.post(f"{base}/generate", headers=headers,
                            json=native_payload, timeout=API_TIMEOUT)
            if r.status_code == 200:
                js = r.json()
                if "text" in js:
                    return (js.get("text") or "").strip()
                if "outputs" in js and js["outputs"]:
                    return (js["outputs"][0].get("text") or "").strip()
                logging.error(f"vLLM native unexpected response: {js}")
            else:
                logging.warning(f"vLLM native {r.status_code}: {r.text}")
        except Exception as e:
            last_err = e
            logging.debug(f"native path failed: {e}")

        logging.error(f"vLLM request failed. Last error: {last_err}")
        return ""


    def _generate_text_with_openai(self, prompt: str, temperature: float, max_tokens: int) -> str:
        last_err = None
        for _ in range(API_RETRY_ATTEMPTS):
            try:
                resp = openai.chat.completions.create(
                    timeout=API_TIMEOUT,
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                last_err = e
                logging.error(f"OpenAI API error: {e}")
                time.sleep(API_RETRY_DELAY)
        return ""

    def _generate_text_with_together(self, prompt: str, temperature: float, max_tokens: int) -> str:
        headers = {
            "Authorization": f"Bearer {self.together_api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for _ in range(API_RETRY_ATTEMPTS):
            try:
                resp = requests.post(
                    "https://api.together.xyz/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=API_TIMEOUT,
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"].strip()
                logging.error(f"Together API error: {resp.status_code} - {resp.text}")
            except Exception as e:
                logging.error(f"Together API request error: {e}")
            time.sleep(API_RETRY_DELAY)
        return ""

def query_llm(text, max_tokens=200, temp=0.5, model="gpt-3.5-turbo", n=3, engine="openai"):
    client = LLMClient(engine=engine, model_name=model)
    outputs: List[str] = []
    for _ in range(max(1, n)):
        out = client.generate_text(text, temperature=temp, max_tokens=max_tokens)
        if out:
            outputs.append(out)
    return outputs

def query_rule(text_list, max_tokens=1500, temperature=0.7, max_try_num=5, model="gpt-3.5-turbo", engine="openai"):
    client = LLMClient(engine=engine, model_name=model)
    result_list: List[str] = []
    for prompt in tqdm(text_list):
        curr_try_num = 0
        success = False
        while curr_try_num < max_try_num and not success:
            try:
                out = client.generate_text(prompt, temperature=temperature, max_tokens=max_tokens)
                if out:
                    result_list.append(out)
                    success = True
                    break
                raise RuntimeError("Empty response from LLM")
            except Exception as e:
                logging.error(e)
                curr_try_num += 1
                if curr_try_num >= max_try_num:
                    result_list.append(-1)
                    break
                time.sleep(API_RETRY_DELAY)
    return result_list

def parse_rank(results):
    rank_dict = {}
    for idx, result in enumerate(results):
        rank_dict[idx] = []
        lines = result.split('\n')
        for line in lines:
            if line.startswith("Rank:"):
                continue
            elif ". " in line:
                rank_item = line.split(". ")[1]
                feature_name = rank_item.split(":")[0].strip()
                # Extract only the feature name before parentheses if they exist
                if "(" in feature_name:
                    feature_name = feature_name.split("(")[0].strip()
                rank_dict[idx].append(feature_name)
    return rank_dict

def parse_relationship(results):
    relationship_dict = {}
    for idx, result in enumerate(results):
        relationship_dict[idx] = {}
        result = result.replace('\\n', '\n')
        lines = result.strip().split('\n')
        current_class = None
        
        # First pass: detect all class labels from correlation statements
        detected_classes = set()
        for line in lines:
            line = line.strip()
            if "correlation with class" in line and "-" in line:
                line_content = line[2:] if line.startswith("-") else line
                if "correlation with class" in line_content:
                    parts = line_content.split(" correlation with class ")
                    if len(parts) > 1:
                        class_label = parts[1].strip()
                        detected_classes.add(class_label)
        
        # Initialize detected classes
        for class_label in detected_classes:
            relationship_dict[idx][class_label] = {}
        
        # Second pass: parse relationships
        for line in lines:
            line = line.strip()
            if "Casual Relationship for class" in line or "Causal Relationship for class" in line:
                current_class = line.split('"')[1]
                if current_class not in relationship_dict[idx]:
                    relationship_dict[idx][current_class] = {}
            elif "-" in line:
                line_content = line[2:]
                if "has a" in line_content and ("correlation with class" in line_content or "relationship with" in line_content):
                    parts = line_content.split(" has a ")
                    feature = parts[0].strip()
                    
                    if "correlation with class" in line_content:
                        # New format: "has a positive/negative correlation with class high/low"
                        relationship_and_class = parts[1].split(" correlation with class ")
                        relationship = relationship_and_class[0].strip()  # "positive" or "negative"
                        class_label = relationship_and_class[1].strip()   # "high" or "low"
                        
                        # Use detected class or current_class
                        target_class = class_label if class_label in relationship_dict[idx] else current_class
                        if target_class is not None:
                            if target_class not in relationship_dict[idx]:
                                relationship_dict[idx][target_class] = {}
                            relationship_dict[idx][target_class][feature] = relationship
                    else:
                        # Original format: "has a relationship"
                        relationship = parts[1].split(" correlation")[0].strip()
                        if current_class is not None:
                            relationship_dict[idx][current_class][feature] = relationship
                elif "is in" in line_content:
                    parts = line_content.split(" is in ")
                    feature = parts[0].strip()
                    if len(parts) > 1:
                        value_list = parts[1].strip().strip('[]').split(', ')
                        if current_class is not None:
                            relationship_dict[idx][current_class][feature] = value_list
    return relationship_dict

def parse_relationship_feature(results, feature_standard_relations):
    parsed_dict = {}
    for (feat1, feat2), relation in feature_standard_relations.items():
        if relation != 'No':
            parsed_dict[(feat1, feat2)] = []
            parsed_dict[(feat2, feat1)] = []
    for result in results:
        result = result.replace('\\n', '\n')
        lines = result.strip().split('\n')
        for line in lines:
            # Handle both formats: "has a relationship with" and "has a positive/negative correlation with class"
            if 'has a' in line and ('relationship with' in line or 'correlation with class' in line):
                line = line.strip('- ').strip('.')
                parts = line.split(' has a ')
                feature1 = parts[0].strip()
                
                if 'relationship with' in line:
                    # Original format: "has a relationship with"
                    relationship_and_feature2 = parts[1].split(' relationship with ')
                    relationship = relationship_and_feature2[0].strip()
                    feature2 = relationship_and_feature2[1].strip()
                else:
                    # New format: "has a positive/negative correlation with class"
                    correlation_and_class = parts[1].split(' correlation with class ')
                    relationship = correlation_and_class[0].strip()  # "positive" or "negative"
                    feature2 = correlation_and_class[1].strip()     # "high" or "low"
                
                feature1 = feature1.replace(' in ', '__')
                feature2 = feature2.replace(' in ', '__')
                if (feature1, feature2) in parsed_dict:
                    parsed_dict[(feature1, feature2)].append(relationship)
                elif (feature2, feature1) in parsed_dict:
                    parsed_dict[(feature2, feature1)].append(relationship)
    return parsed_dict

def parse_llm_results(label_list, golden_features_nm, results):
    parsed_results = {label: {feature: [] for feature in golden_features_nm} for label in label_list}

    for result in results:
        # Handle both literal \n and actual newlines -> replace literal \n with actual newlines
        result = result.replace('\\n', '\n')
        lines = result.strip().split('\n')
        current_label = None
        features_processed = {label: set() for label in label_list}

        for line in lines:
            line = line.strip().strip('- ')

            if line.startswith("Condition for class"):
                current_label = line.split('"')[1]

            elif 'less than or equal to' in line or 'less than' in line or 'greater than or equal to' in line or 'greater than' in line:
                feature, condition = line.split(' is ', 1)
                feature = feature.strip()
                if 'less than or equal to' in condition:
                    value = re.search(r'\d+(?:\.\d+)?', condition.replace('less than or equal to', '')).group()
                elif 'less than' in condition:
                    value = re.search(r'\d+(?:\.\d+)?', condition.replace('less than', '')).group()
                elif 'greater than or equal to' in condition:
                    value = re.search(r'\d+(?:\.\d+)?', condition.replace('greater than or equal to', '')).group()
                elif 'greater than' in condition:
                    value = re.search(r'\d+(?:\.\d+)?', condition.replace('greater than', '')).group()

                try:
                    value = float(value)
                except ValueError:
                    pass

                parsed_results[current_label][feature].append(value)
                features_processed[current_label].add(feature)

            elif 'is in' in line:
                feature, categories = line.split(' is in ', 1)
                feature = feature.strip()
                categories = categories.strip('[]').replace("'", "").split(',')
                categories = [category.strip() for category in categories]

                parsed_results[current_label][feature].append(categories)
                features_processed[current_label].add(feature)

            elif 'is' in line:
                feature, categories = line.split(' is ', 1)
                feature = feature.strip()
                categories = categories.strip('[]').replace("'", "").split(',')
                categories = [category.strip() for category in categories]

                parsed_results[current_label][feature].append(categories)
                features_processed[current_label].add(feature)

        for label in label_list:
            for feature in golden_features_nm:
                if feature not in features_processed[label]:
                    parsed_results[label][feature].append(None)

    return parsed_results

def get_result(llm, data, rule, template, mode, temp, engine="openai"):
    llm_dir = 'gpt/' if 'gpt' in llm.lower() else ""
    rule_file_name = f'./{"diagnose" if mode=="diagnose" else "evaluate"}/{data}/{llm_dir}{rule}.out'


    if not os.path.isfile(rule_file_name):
        if mode == 'diagnose':
            results = query_llm(template, max_tokens=200, temp=temp, model=llm, engine=engine)
        else:
            results = query_rule(template, max_tokens=1500, model=llm, engine=engine)
        if results:
            with open(rule_file_name, 'w') as f:
                f.write(_DIVIDER.join(results))
        else:
            logging.error("LLM returned empty results; skip writing cache for %s", rule_file_name)
    else:
        logging.info("Found cached file")
        with open(rule_file_name, 'r') as f:
            total_rules_str = f.read().strip()
            results = total_rules_str.split(_DIVIDER)
    return results

def parse_result(results, level, setup_info):
    logging.info(f"Parsing results for {level}")
    if level == 'lv1':
        parsed_result = parse_rank(results)
    elif level == 'lv2':
        parsed_result = parse_relationship(results)
    else:
        parsed_result = parse_llm_results(setup_info["label_list"], setup_info["golden_variables"], results)
    
    return parsed_result

def parse_rules(result_texts, label_list=[]):
    total_rules = []
    splitters = ["onditions for class", "onditions for the","For class", "Class \"", "Class "]
    for text in result_texts:
        if not any(splitter in text for splitter in splitters):
            continue
        used_splitter = next(splitter for splitter in splitters if splitter in text)
        splitted = text.split(used_splitter)

        rule_raws = splitted[1:]
        rule_dict = {}
        for rule_raw in rule_raws:
            class_name = rule_raw.split(":")[0].strip(" .'").strip(' []"')
            class_name = class_name.split("\n")[0].strip()
            class_name = class_name.rstrip('"*')
            class_name = class_name.replace(" class:", "").replace(" class", "").strip()
            class_name = class_name.replace(" conditions:", "").replace(" conditions", "").strip()
            class_name = class_name.strip('"')
            rule_parsed = []
            for line in rule_raw.split("\n")[1:]:
                line = line.strip()
                if line.startswith("- "):
                    rule_parsed.append(line[2:])
            rule_dict[class_name] = rule_parsed
        total_rules.append(rule_dict)
    return total_rules

def normalize_conditions(rules):
    normalized_rules = []
    for rule in rules:
        rule = re.sub(r"less than or equal to", "<=", rule, flags=re.IGNORECASE)
        rule = re.sub(r"greater than or equal to", ">=", rule, flags=re.IGNORECASE)
        rule = re.sub(r"less than", "<", rule, flags=re.IGNORECASE)
        rule = re.sub(r"greater than", ">", rule, flags=re.IGNORECASE)

        rule = re.sub(r"([<>]=?)\s*\[([0-9]+)\]", r"\1 \2", rule)
        normalized_rules.append(rule)
    return normalized_rules

def match_feature_case(rules, golden_features):
    matched_rules = []
    for rule in rules:
        for feature in golden_features:
            rule = re.sub(rf"\b{feature}\b", feature, rule, flags=re.IGNORECASE)
        matched_rules.append(rule)
    return matched_rules

def normalize_parsed_rules(parsed_rules, setup_info):
    normalized_parsed_rules = []
    for idx, rule_dict in enumerate(parsed_rules):
        normalized_rule_dict = {}

        for class_label in setup_info["label_list"]:
            if class_label not in rule_dict:
                continue

            rules = rule_dict[class_label]
            rules = normalize_conditions(rules)
            rules = match_feature_case(rules, setup_info['golden_variables'])
            rules = [rule.strip('.') for rule in rules]

            normalized_rule_dict[class_label] = rules
        normalized_parsed_rules.append(normalized_rule_dict)

    return normalized_parsed_rules

def query_parse_llm(text_list, feat_num, max_tokens=30, temperature=0, model="gpt-4o", engine="openai"):
    client = LLMClient(engine=engine, model_name=model)
    result_list: List[str] = []
    max_try_num = max(1, feat_num)
    for prompt in tqdm(text_list):
        curr_try_num = 0
        while curr_try_num < max_try_num:
            try:
                out = client.generate_text(prompt, temperature=temperature, max_tokens=max_tokens)
                if out:
                    result_list.append(out)
                    break
                raise RuntimeError("Empty response from LLM")
            except Exception as e:
                logging.error(f"Error: {e}")
                curr_try_num += 1
                time.sleep(API_RETRY_DELAY)
    return result_list

def prompt_for_function(data, parsed_rules, feature_desc, func_rule, feat_num, engine="openai"):
    saved_file_name = f'./evaluate/{data}/{func_rule}.out'

    if not os.path.isfile(saved_file_name):
        function_file_name = './templates/ask_for_function.txt'
        fct_strs_all = []
        for parsed_rule in tqdm(parsed_rules):
            fct_templates = generate_prompt_for_function(
                            parsed_rule, feature_desc, function_file_name
                        )
            fct_results = query_parse_llm(fct_templates, feat_num, max_tokens=1500, temperature=0, engine=engine)
            fct_strs = [fct_txt.split('<start>')[1].split('<end>')[0].strip() for fct_txt in fct_results]
            fct_strs_all.append(fct_strs)

            with open(saved_file_name, 'w') as f:
                total_str = _VERSION.join([_DIVIDER.join(x) for x in fct_strs_all])
                f.write(total_str)
    else:
        with open(saved_file_name, 'r') as f:
            total_str = f.read().strip()
            fct_strs_all = [x.split(_DIVIDER) for x in total_str.split(_VERSION)]
    return fct_strs_all

