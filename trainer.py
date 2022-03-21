# ！/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = "L"

from config import *
import os
from argparse import ArgumentParser
from collections import defaultdict
from logging import Logger
from typing import Optional, Union, Callable, Tuple, Dict
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import math
import sys
from sklearn.metrics import accuracy_score, f1_score, recall_score

from tricks import FGM


class Trainer:
    def __init__(self,
                 model: Union[PreTrainedModel, nn.Module] = None,
                 args: Optional[ArgumentParser] = None,
                 train_loader: Optional[DataLoader] = None,
                 eval_loader: Optional[Dict] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 model_init: Callable[[], PreTrainedModel] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 logger=None):
        self.model = model
        self.args = args
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.tokenizer = tokenizer
        self.model_init = model_init
        self.optimizer = optimizers[0]
        self.scheduler = optimizers[1]
        self.logger = logger
        # 实例化模型
        self.device = args.device
        self.model.to(args.device)

    def train_one_epoch(self, args, model, optimizer, data_loader, device, epoch, tb_writer, fgm=None):
        model.train()
        mean_loss = torch.zeros(1).to(device)
        optimizer.zero_grad()

        data_loader = tqdm(data_loader, dynamic_ncols=True, file=sys.stdout)
        for iteration, data in enumerate(data_loader):
            batch, labels = data
            batch, labels = batch.to(device), labels.to(device)
            output = model(batch, labels)
            pred, total_loss = output.logits, output.total_loss
            total_loss.retain_grad()
            total_loss.backward()

            if args.use_fgm:
                fgm.attack(args.fgm_eps, 'pretrain_model.embeddings.word_embeddings.weight')  # embedding被修改了
                output = model(batch, labels)
                pred, total_loss = output.logits, output.total_loss
                total_loss.backward()  # 反向传播，在正常的grad基础上，累加对抗训练的梯度
                fgm.restore(args.fgm_eps, 'pretrain_model.embeddings.word_embeddings.weight')  # 恢复Embedding的参数

            mean_loss = (mean_loss * iteration + total_loss.detach()) / (iteration + 1)  # update mean losses

            metrics = self.metric_function(pred, labels)
            mean_metrics = {}
            for key, value in metrics.items():
                mean_metrics[key] = value * iteration + (sum(value.values()) / len(value)) / (iteration + 1)
            # 打印平均loss
            if iteration % 50 == 0:
                data_loader.desc = "[epoch {}]".format(epoch) + " ".join(
                    ["{} {}".format(key, round(metric.item(), 3)) for key, metric in mean_metrics.items()])

            if not torch.isfinite(total_loss):
                print('WARNING: non-finite loss, ending training ', total_loss)
                sys.exit(1)

            optimizer.step()
            optimizer.zero_grad()
            tags = ["mean_loss", "learning_rate"]

            # tensorboard可视化，for循环加入或者直接传入字典
            for tag, value in zip(tags, [mean_loss.item(), optimizer.param_groups[0]["lr"]]):
                tb_writer.add_scalars('Train/', {tag: value}, iteration)
            tb_writer.add_scalars('Train/', metrics, iteration)

        return mean_loss.item(), mean_metrics

    def save_model(self, model, optimizer, scheduler, epoch, best_f1, num_params, model_name):
        checkpoint = {
            'model_state_dict': model.state_dict(),  # *模型参数
            'optimizer_state_dict': optimizer.state_dict(),  # *优化器参数
            'scheduler_state_dict': scheduler.state_dict(),  # *scheduler
            'epoch': epoch,
            'best_val_mae': best_f1,
            'num_params': num_params
        }
        torch.save(checkpoint, os.path.join(self.args.model_output_path, model_name))
        self.logger.info('save model {} successed......'.format(model_name[:-3]))

    def train(self):
        self.logger.info("********** Train examples: {}".format(self.train_dataloader.dataset.num_examples))
        self.logger.info("********** Train batches per epoch: {}".format(self.train_dataloader.dataset.num_batches))
        if self.val_loader:
            self.logger.info("********** Val examples: {}".format(self.val_dataloader.dataset.num_examples))
            self.logger.info("********** Val batches per epoch: {}".format(self.val_dataloader.dataset.num_batches))
        self.logger.info("********** Model version: {}".format(self.args.model_version))
        # 写入日志
        self.logger.info('start training......\n')

        tb_writer = SummaryWriter(log_dir=os.path.join(self.args.log_output_path, 'tensorboard_logs'))
        # 将模型写入tensorboard
        # init_input,init_label=next(iter(train_dataloader))
        # init_input = torch.zeros((1, 3, 224, 224), device=args.device)
        # tb_writer.add_graph(model, init_input)

        best_f1 = 0  # 最佳模型的指标
        patient = 0  # 早停patient=5

        # 对抗训练
        fgm = None
        if self.args.use_fgm: fgm = FGM(self.model)
        for epoch in range(self.args.epochs):
            mean_loss, mean_metrics = self.train_one_epoch(args=self.args,
                                                           model=self.model,
                                                           optimizer=self.optimizer,
                                                           data_loader=self.train_dataloader,
                                                           device=self.device,
                                                           epoch=epoch,
                                                           tb_writer=tb_writer,
                                                           fgm=fgm)

            # update learning rate
            self.scheduler.step()

            self.logger.info('*********** val ')
            # validate
            eval_metrics = self.eval(self.eval_dataloader, self.device, self.logger,
                                     self.args.log_output_path, self.args.log_output_path, 'val')

            val_f1 = eval_metrics['micro-f1']
            # tensorboard可视化
            tb_writer.add_scalars('Validation', eval_metrics, epoch)
            num_params = sum(p.numel() for p in self.model.parameters())
            if epoch % self.args.save_epoch == 0:
                self.save_model(self.model, self.optimizer, self.scheduler, epoch, best_f1, num_params,
                                model_name='checkpoint-%d.pt' % epoch)
            # 保存最佳模型
            if best_f1 < val_f1:
                best_f1 = val_f1
                mean_f1 = mean_metrics['f1']
                self.logger.info('best model in %d epoch, train mean f1: %.2f ' % (epoch, mean_f1))
                self.logger.info('best model in %d epoch, validation f1: %.2f ' % (epoch, val_f1))
                self.save_model(self.model, self.optimizer, self.scheduler, epoch, best_f1, num_params,
                                model_name='best_checkpoint.pt')
                patient = 0
            else:
                patient += 1
                print("Counter {} of {}".format(patient,self.args.earlystop_patient))
            if patient > self.args.earlystop_patient:
                self.logger.info(
                    "Early stopping [epoch %i]" % epoch + "with best_f1: %2f" % best_f1 + "and val_f1: %2f" % val_f1 + "...")
                break

        self.logger.info("********** load best model weights")
        self.model.load_state_dict(
            torch.load(os.path.join(self.args.model_output_path, 'best_checkpoint.pt'))['model_state_dict'])
        self.logger.info("********** print best result")

        # validate
        self.eval(self.eval_dataloader, self.device, self.logger,
                  self.args.csv_output_path, 'val')
        return self.model

    def metric_function(self, y_true, y_pred):
        # 这里定义一些常见的评价指标
        metrics = {}
        metrics['acc'] = accuracy_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred)
        # 有的时候是多分类的标签，求micro或者macro
        # recall_score(y_true,y_pred,average='macro')
        # f1_score(y_true, y_pred,average='macro')
        return metrics

    @torch.no_grad()
    def eval(self, data_loader, csv_output_path, sce):
        self.logger.info('*********** val ' + sce + 'dataset')
        self.model.eval()
        # 可以把模型预测的输出写成csv
        csv_writer = open(csv_output_path, 'w')

        # 打印验证进度
        data_loader = tqdm(data_loader, desc="predict...", dynamic_ncols=True, file=sys.stdout)
        res = []
        y_true = []
        for step, batch in enumerate(data_loader):
            inputs, labels = batch
            output = self.model(inputs.to(self.device))
            pred = output.logits
            res.extend(pred)
            y_true.extend(labels)
            # 可以把模型预测的输出写成csv
            csv_writer.write('...')

        csv_writer.close()
        metrics = self.metric_function(y_true, res)
        return metrics

    @torch.no_grad()
    def predict(self, data_loader, csv_output_path):
        self.model.eval()
        self.logger.info('*********** predict ')
        data_loader = tqdm(data_loader, desc="predict...", dynamic_ncols=True, file=sys.stdout)
        res = []
        csv_writer = open(csv_output_path, 'w')
        for step, batch in enumerate(data_loader):
            output = self.model(batch.to(self.device))
            pred = output.logits
            res.extend(pred)
            csv_writer.write('...')

        csv_writer.close()
        return res
