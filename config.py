#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration module for FE Evaluation

This module loads configuration from environment variables and provides
default values for various settings.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
VLLM_SERVER_URL = os.getenv('VLLM_SERVER_URL', 'http://localhost:8000')

# Model Settings
DEFAULT_LLM_ENGINE = os.getenv('DEFAULT_LLM_ENGINE', 'together')
DEFAULT_LLM_MODEL = os.getenv('DEFAULT_LLM_MODEL', 'gpt-3.5-turbo')

# Evaluation Settings
DEFAULT_DATA = os.getenv('DEFAULT_DATA', 'adult')
DEFAULT_MODE = os.getenv('DEFAULT_MODE', 'evaluate')
DEFAULT_LEVEL = os.getenv('DEFAULT_LEVEL', 'lv2')
DEFAULT_SHOT = int(os.getenv('DEFAULT_SHOT', '4'))
DEFAULT_SAMPLING_METHOD = os.getenv('DEFAULT_SAMPLING_METHOD', 'random')
DEFAULT_DESC_METHOD = os.getenv('DEFAULT_DESC_METHOD', 'detail')
DEFAULT_SWAP = os.getenv('DEFAULT_SWAP', 'None')
DEFAULT_SEED = int(os.getenv('DEFAULT_SEED', '0'))
DEFAULT_TEMP = float(os.getenv('DEFAULT_TEMP', '0.5'))

DEFAULT_NUM_FEATURE_SET = int(os.getenv('DEFAULT_NUM_FEATURE_SET', '10'))
DEFAULT_NUM_FEATURE = int(os.getenv('DEFAULT_NUM_FEATURE', '7'))
DEFAULT_OUTPUT_DIR = os.getenv('DEFAULT_OUTPUT_DIR', 'logs')