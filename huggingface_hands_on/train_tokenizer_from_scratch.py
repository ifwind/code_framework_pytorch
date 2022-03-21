import joblib
from config import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from config import *
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel

tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
from tokenizers.trainers import BpeTrainer, WordLevelTrainer

# 从头开始预训练一个tokenizer

# 加入一些特殊字符
trainer = WordLevelTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# 空格分词器
from tokenizers.pre_tokenizers import Whitespace

tokenizer.pre_tokenizer = Whitespace()

import glob

data_files = glob.glob(r'text_*.csv')
# 保存语料库文件
tokenizer.train(data_files, trainer)  # [data_file]
tokenizer.mask_token = '[MASK]'
from tokenizers.processors import TemplateProcessing

tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)
tokenizer.save("../tokenizer.json")
