# ！/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = "L"

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from torchcrf import CRF
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, BertModel
from transformers.file_utils import ModelOutput
import numpy as np
from config import *


@dataclass
class MyModelOutput(ModelOutput):
    total_loss: Optional[torch.FloatTensor] = None
    losses: Optional[Dict] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class OutputLayer(nn.Module):
    def __init__(self, config, num_classify, num_ner):
        super(OutputLayer, self).__init__()
        self.config = config
        self.act=nn.GELU
        self.linear_list=nn.ModuleList([nn.Linear(self.config.hidden_size,self.config.hidden_size) for _ in range(3)])
        self.output_layer = nn.ModuleDict(
            {'classify': nn.Linear(self.config.hidden_size, num_classify + 1),
             'ner': nn.Linear(self.config.hidden_size, num_ner + 1)})

    #   self._init_weights(self.output_layer)
    # # 模型参数初始化
    # def _init_weights(self, module):
    #     """Initialize the weights."""
    #     if isinstance(module, nn.Linear):
    #         # Slightly different from the TF version which uses truncated_normal for initialization
    #         # cf https://github.com/pytorch/pytorch/pull/5617
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)

    def forward(self, cls_emb, tokens_emb):
        output = {}
        for linear in self.linear_list:
            cls_emb=linear(cls_emb)
        output['classify_logits'] = self.output_layer['layer1'](cls_emb)
        output['ner_logits'] = self.output_layer['layer2'](tokens_emb)
        return output


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, y_pred, y_true):
        lossf = F.cross_entropy
        losses = {}
        total_loss = 0

        losses['classify'] = lossf(y_pred, y_true)
        losses['ner'] = lossf(y_pred, y_true)
        total_loss += losses['']

        return total_loss, losses


class MyModel(BertModel):
    def __init__(self, config, num_intent, num_slot, cache_dir='./pretrain_model_path/'):
        super().__init__(config)
        self.config = config
        self.num_intent = num_intent
        self.num_slot = num_slot

        self.pretrain_model = AutoModel.from_pretrained(config.name_or_path, cache_dir=cache_dir)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        # self.pretrain_model.embeddings.word_embeddings.weight.retain_grad()
        self.output_layer = OutputLayer(config, num_intent, num_slot)
        # Initialize weights and apply final processing
        self.post_init()
        self.crf = CRF(num_slot, batch_first=True)
        self.loss_function = MyLoss()

    def forward(self, inputs, labels=None, output_attentions=False, output_hidden_states=False):
        outputs = self.pretrain_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids'],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output, sequence_output = outputs[1], outputs[0]
        if labels is not None:
            pooled_output = self.dropout(pooled_output)

        output = self.output_layer(pooled_output, sequence_output, inputs['classify_label'])
        intent_logits, slot_logits = output['classify_logits'], output['ner_logits']

        total_loss, losses = None, None
        loss_mask = inputs['token_type_ids'].gt(0)
        if labels is not None:
            slot_loss = self.crf(slot_logits, labels['slots'], loss_mask)
            intent_loss = nn.CrossEntropyLoss(intent_logits, labels['intent'])
            total_loss = slot_loss + intent_loss

        sequence_of_tags = self.crf.decode(slot_logits, mask=loss_mask)
        return MyModelOutput(
            total_loss=total_loss,
            losses=losses,
            logits={'classify_logits': intent_logits, 'ner_tags': sequence_of_tags},
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 如果需要为预训练模型添加新的token
    def resize_token_embeddings(self, re_token_size):
        self.pretrain_model.resize_token_embeddings(re_token_size)
