# ！/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = "L"

from transformers import BertTokenizer, AutoConfig, Trainer
from model_utils import MyModel
from logging_config import get_logger
import math
import argparse
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import sys
from config import *
from data_loader import *
from trainer import Trainer
from tqdm import tqdm
from tricks import *


def init_train(args):
    if args.log_output_path == "":
        args.log_output_path = "./logs/{}/".format(args.model_version)
    args.log_output_file = args.log_output_path + args.log_output_file
    if args.model_output_path == "":
        args.model_output_path = "./model_file/{}/".format(args.model_version)
    if not os.path.exists(args.log_output_path):
        os.makedirs(args.log_output_path)
    if not os.path.exists(args.model_output_path):
        os.makedirs(args.model_output_path)

    if args.mem_size not in [1, 2, 3, 4, 5]:
        raise ValueError("Invalid mem size")

    logger = get_logger(args.log_output_file)
    logger.info("********** GPU device: {}".format(args.device))

    logger.info("********** Data path: {}".format(args.data_path))
    logger.info("********** Learning rate: {}".format(args.lr))
    logger.info("********** Batch size: {}".format(args.batch_size))

    logger.info("********** Epochs: {}".format(args.epochs))
    logger.info("********** Mem size: {}".format(args.mem_size))

    logger.info("********** Do train: {}".format(args.do_train))
    logger.info("********** Use FGM: {}".format(args.use_fgm))
    logger.info("********** FGM eps: {}".format(args.fgm_eps))

    return logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--warm_up_epochs', type=int, default=5)
    parser.add_argument('--earlystop_patient', type=int, default=5)
    parser.add_argument('--save_epoch', type=float, default=3)
    parser.add_argument('--pretrain_model_path', type=str, default="Resources/bert-base-chinese")
    # --freeze-layers #如果是True表示冻结除了全连接层以外的所有层的参数，在导入一些预训练的模型可以使用，可以加快模型训练
    parser.add_argument('--freeze_layers', type=bool, default=False)
    parser.add_argument('--device', default='cpu', help='device id (i.e. cuda:0 or 0,1 or cpu)')
    parser.add_argument('--model_output_path', type=str, default="")
    parser.add_argument('--log_output_path', type=str, default="")
    parser.add_argument('--log_output_file', type=str, default="out.log")
    # 数据设置
    parser.add_argument("--data_path", type=str, default="./dataset/business_service/multi_task/hf_v1/0",
                        help="data path")
    # flags.DEFINE_string("data_path", "./dataset", "data path")
    # 模型定义参数
    parser.add_argument("--mem_size", type=int, default=5, help="mem size")
    # 训练参数
    parser.add_argument("--random_seed", type=int, default=666, help="random seed")
    parser.add_argument("--do_train", type=bool, default=True, help="do train")
    parser.add_argument("--use_fgm", type=bool, default=True, help="use fgm")
    parser.add_argument("--fgm_eps", type=float, default=0.5, help="fgm eps")
    parser.add_argument("--model_version", type=str, default='0', help="model version")
    args = parser.parse_args()

    # 设置随机种子，为了模型效果可复现
    seed_torch(args.random_seed)
    # 日志设置
    logger = init_train(args)

    # os.environ['TRANSFORMERS_CACHE'] = './pretrain_model_path/'
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path, cache_dir='./pretrain_model_path/')
    # 如果需要增加特殊字符
    # special_tokens_dict = {'additional_special_tokens': ["[PAD]"]}
    # tokenizer.add_special_tokens(special_tokens_dict)
    # debug模式下用少量的数据
    if args.debug:
        train_dataloader = MyDataLoader(args.data_path, tokenizer, args.batch_size, args.random_seed, 'debug')
        val_dataloader = MyDataLoader(args.data_path, tokenizer, args.batch_size, args.random_seed, 'debug')
    else:
        train_dataloader = MyDataLoader(args.data_path, tokenizer, args.batch_size, args.random_seed, 'train')
        dev_dataloader = MyDataLoader(args.data_path, tokenizer, args.batch_size, args.random_seed, 'val')
    config = AutoConfig.from_pretrained(args.pretrain_model_path, cache_dir='./pretrain_model_path/')
    model = MyModel(config, cache_dir='./pretrain_model_path/')
    model.resize_token_embeddings(len(tokenizer))
    # cosine lr_lambda
    lambda_cosine = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    # warm up lr_lambda
    warm_up_with_cosine_lr = lambda epoch: epoch / args.warm_up_epochs if epoch <= args.warm_up_epochs else 0.5 * (
            math.cos((epoch - args.warm_up_epochs) / (args.epochs - args.warm_up_epochs) * math.pi) + 1)

    # 是否冻结权重
    if args.freeze_layers:
        print("freeze layers except fc layer.")
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=args.lr, weight_decay=0.005)

    # 全局warm up
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    # cosine学习率下降
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_cosine)

    # 不同模型层不同学习率
    # 预训练层-小学习率
    # 微调层-warm up学习率
    # ignored_params = list(map(id, model.output_heads.parameters()))
    # base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    # optimizer = optim.Adam([
    #         {'params': base_params},
    #         {'params': model.output_heads.parameters(), 'lr': 0.001}], 5e-5,weight_decay=1e-4)
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda_cosine,warm_up_with_cosine_lr])

    trainer = Trainer(model=model,
                      args=args,
                      train_loader=train_dataloader,
                      eval_loader=dev_dataloader,
                      tokenizer=tokenizer,
                      model_init=None,
                      optimizers=(optimizer, scheduler),
                      logger=logger)

    model = trainer.train()

    # 如果有开发集/新的验证集之类的也可以这样验证
    val_dataloader = '...'
    trainer.eval(val_dataloader, csv_output_path='...', sce='val')
