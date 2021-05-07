# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 14:50:18 2021

"""
import torch
import os

class StrategizedTokenizerDataset(torch.utils.data.Dataset):
    def __init__(self, datadir='../pretraining_data'):
        self.encodings = {key: torch.tensor([], dtype=torch.long)  for key in ['input_ids', 'attention_mask']}
        self.labels = torch.tensor([], dtype=torch.long)
        self.datadir = datadir

    def __getitem__(self, idx):
        item = {key: val[idx].long() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx].long()
        return item
    
    def __len__(self):
        return len(self.labels)
    
    def populate(self, book_list = [16968, 1741]):
        
        #TODO make dynamic based on desired datasize
        #Book list will change based on what is desired, current default is examplary.
        
        for book_id in book_list:
            if os.path.exists(os.path.join(self.datadir, str(book_id))):
                saved_encodings = torch.load(os.path.join(self.datadir, str(book_id), "tensor_file.pt"))
                self.encodings = {key: torch.cat((self.encodings[key], saved_encodings[key])) for key,val in saved_encodings.items() if key != 'labels'}
                self.labels = torch.cat((self.labels, saved_encodings['labels']))
            else:
                raise FileExistsError("{} does not exist. ".format(os.path.join(self.datadir, str(book_id))))
        print('Loaded books: ', book_list)
                
                
class DefaultTokenizerDataset(torch.utils.data.Dataset):
    def __init__(self, datadir='../pretraining_data'):
        self.encodings = {key: torch.tensor([], dtype=torch.long)  for key in ['input_ids', 'attention_mask']}
        self.labels = torch.tensor([], dtype=torch.long)
        self.datadir = datadir
    
        
        
    
    
    
    

