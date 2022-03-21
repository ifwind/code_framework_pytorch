# ！/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = "L"

# 继续预训练+微调
from transformers import DataCollatorForLanguageModeling, Trainer, BertTokenizer, \
    TrainingArguments, AutoModelForPreTraining
import pickle
from datasets import Dataset, load_from_disk, concatenate_datasets
import numpy as np

if __name__ == '__main__':
    # 数据预处理
    pretrain_model_path = "./pretrain_model_path/albert_chinese_tiny"
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path, cache_dir='./pretrain_model_path/')

    # 数据准备
    # 数据加载
    raw_data = pickle.load(open(r'./dataset/raw_data.pkl', 'rb'))
    p_data = {'sentenceA': [], 'sentenceB': []}
    for key, s_pair in raw_data.items():
        p_data['sentenceA'].extend(s_pair['sentenceA'])
        p_data['sentenceB'].extend(s_pair['sentenceB'])

    res = Dataset.from_dict(p_data)


    # 数据集预处理函数
    def preprocess_function(examples):
        return tokenizer(examples['sentenceA'], examples['sentenceB'], truncation=True,
                         max_length=min(tokenizer.max_model_input_sizes.values()))


    # 数据预处理
    cache_dir = './dataset/data_for_pretrain'
    encoded_dataset = res.map(preprocess_function, batched=True)
    encoded_dataset.save_to_disk(cache_dir)

    dataset = []  # 这里是为了演示如果有多个dataset切片怎么加载并且拼成一个dataset
    for i in range(0, 1):
        cache_dir = './dataset/data_for_pretrain'
        encoded_dataset = load_from_disk(cache_dir)
        dataset.append(encoded_dataset)
    encoded_dataset = concatenate_datasets(dataset)

    encoded_dataset = encoded_dataset.remove_columns(['sentenceA', 'sentenceB'])
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15  # mlm表示是否使用masked language model；mlm_probability表示mask的几率
    )
    model = AutoModelForPreTraining.from_pretrain(pretrain_model_path)

    training_args = TrainingArguments(
        output_dir="./pre_train",
        overwrite_output_dir=True,
        num_train_epochs=2,  # 训练epoch次数
        per_gpu_train_batch_size=64,  # 训练时的batchsize
        save_steps=10_000,  # 每10000步保存一次模型
        save_total_limit=2,  # 最多保存两次模型
        prediction_loss_only=True,
    )

    trainer = Trainer(  # 注意这里的trainer是transformers里面的
        model=model,
        args=training_args,
        data_collator=data_collator,  # 数据收集器在这里
        train_dataset=encoded_dataset  # 注意这里选择的是预处理后的数据集
    )
