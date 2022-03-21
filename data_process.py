from collections import defaultdict
from transformers import BertTokenizer
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import joblib
from config import *


def preprocess(data_path, random_state):
    rawdata = pd.read_csv(data_path)  # 读取原始数据
    rawdata.reset_index(inplace=True, drop=True)

    X = list(rawdata.index)
    y = rawdata['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                        stratify=y,
                                                        random_state=random_state)  # stratify=y表示分层抽样，根据不同类别的样本占比进行抽样
    # 如果担心随机种子不起作用可以保存下train/val/dev的索引值
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.1,
                                                      stratify=y, random_state=random_state)
    test_data = {'X_test': X_test, 'y_test': y_test}
    joblib.dump(test_data, 'test_index.pkl')
    train_data = {'X_train': X_train, 'y_train': y_train}
    joblib.dump(train_data, 'train_index.pkl')
    dev_data = {'X_dev': X_dev, 'y_dev': y_dev}
    joblib.dump(dev_data, 'dev_index.pkl')

    train_x = rawdata.loc[train_data['X_train']]['words']
    train_y = rawdata.loc[train_data['X_train']]['label'].values
    test_x = rawdata.loc[test_data['X_test']]['words']
    test_y = rawdata.loc[test_data['X_test']]['label'].values
    dev_x = rawdata.loc[train_data['X_train']]['words']
    dev_y = rawdata.loc[train_data['X_train']]['label'].values

    joblib.dump({'train_x': train_x, 'train_y': train_y}, 'train_datapieces.pkl')
    joblib.dump({'dev_x': dev_x, 'dev_y': dev_y}, 'dev_datapieces.pkl')
    joblib.dump({'test_x': test_x, 'test_y': test_y}, 'test_datapieces.pkl')


def data_propress():
    sces = ['train', 'dev', 'test']

    for sce in sces:
        data = joblib.load('../dataset/{}_datapieces.pkl'.format(sce))

        new_data = defaultdict(list)
        valid_columns = []
        for idx, sample in enumerate(data):
            for key in valid_columns:
                new_data[key].append(sample[key])

        tokenizer = BertTokenizer.from_pretrained("pretrain_model_path/bert-base-chinese")

        # 如果需要添加新的token
        # special_tokens_dict = {'additional_special_tokens': [""]}
        # tokenizer.add_special_tokens(special_tokens_dict)

        # 数据集预处理函数
        def preprocess_function(examples):
            for i in range(len(examples['sentenceA'])):
                pass
            return tokenizer(examples['sentenceA'], examples['sentenceB'], truncation=True,
                             max_length=min(tokenizer.max_model_input_sizes.values()))

        res = Dataset.from_dict(new_data)
        # 数据预处理
        cache_dir = './dataset/{}_dataset'.format(sce)
        encode_dataset = res.map(preprocess_function, batched=True)
        encode_dataset.save_to_disk(cache_dir)


if __name__ == '__main__':
    preprocess(data_path='', random_state=42)
    data_propress()
