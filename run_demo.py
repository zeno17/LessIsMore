# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:22:31 2021

@author: s145733
"""

#Preprocessing code
from cleaner_utils import super_cleaner
from preprocessing_utils import book_to_sentences, whole_word_MO_tokenization_and_masking
from preprocessing_utils import MODataset
from gutenberg.acquire import load_etext

#Training code
from transformers import BertConfig
from transformers import BertForMaskedLM
from transformers import BertTokenizer
from transformers import AdamW
from transformers import Trainer, TrainingArguments

#General imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
import re
import json
import torch

if __name__ == "__main__":
    cleaned_book = super_cleaner(load_etext(16968), -1, verify_deletions=False)
    sentences = book_to_sentences(cleaned_book)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    nlp = spacy.load("en_core_web_sm")

    inputs = whole_word_MO_tokenization_and_masking(tokenizer, nlp, sentences[0])
    train_dataset = MODataset(inputs)
    
    model = BertForMaskedLM(config=BertConfig())
    
    training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total # of training epochs
    per_device_train_batch_size=1,  # batch size per device during training
    #per_device_eval_batch_size=256,   # batch size for evaluation
    learning_rate=1e-5,     
    logging_dir='./logs',            # directory for storing logs
    
    )
    
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=None            # evaluation dataset
    )
    
    trainer.train()
    
    