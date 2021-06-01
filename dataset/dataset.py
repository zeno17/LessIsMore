# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 14:50:18 2021

"""
import torch
import os
import pickle
from transformers import BertTokenizer

class StrategizedTokenizerDataset(torch.utils.data.Dataset):
    def __init__(self, datadir='../pretraining_data', max_seq_length=8):
        self.encodings = {key: torch.tensor([], dtype=torch.long)  for key in ['input_ids', 'attention_mask']}
        self.labels = torch.tensor([], dtype=torch.long)
        self.datadir = datadir
        self.max_seq_length = max_seq_length

    def __getitem__(self, idx):
        item = {key: val[idx].long() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx].long()
        return item
    
    def __len__(self):
        return len(self.labels)
    
    def populate(self, book_list = [16968, 1741]):
        for book_id in book_list:
            if os.path.exists(os.path.join(self.datadir, str(book_id))):
                saved_encodings = torch.load(os.path.join(self.datadir, str(book_id), "tensors_" + str(self.max_seq_length) + ".pt"))
                self.encodings = {key: torch.cat((self.encodings[key], saved_encodings[key])) for key,val in saved_encodings.items() if key != 'labels'}
                self.labels = torch.cat((self.labels, saved_encodings['labels']))
            else:
                raise FileExistsError("{} does not exist. ".format(os.path.join(self.datadir, str(book_id))))
        print(self.max_seq_length, 'Loaded books: ', book_list)
                


class DefaultTokenizerDataset(torch.utils.data.Dataset):
    def __init__(self, datadir='../pretraining_data', max_seq_length=8):
        self.datadir = datadir
        self.max_seq_length = max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        self.sentences = []
        self.encodings = None
    
    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx): 
        return self.encodings[idx]
    
    def populate(self, book_list = [16968, 1741]):
        for book_id in book_list:
            if os.path.exists(os.path.join(self.datadir, str(book_id))):
                with open(os.path.join(self.datadir, str(book_id), "sentences_" + str(self.max_seq_length) + ".pkl"), 'rb') as f:
                    saved_sentences = pickle.load(f)
                self.sentences += saved_sentences
                
            else:
                raise FileExistsError("{} does not exist. ".format(os.path.join(self.datadir, str(book_id))))
                
        self.encodings = self.tokenizer(self.sentences, add_special_tokens=True, truncation=True, max_length=self.max_seq_length)["input_ids"]
        self.encodings = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.encodings]
        print(self.max_seq_length, 'Loaded books: ', book_list)

        
        
    
    
    
    

