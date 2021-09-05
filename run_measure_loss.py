# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:53:43 2021
"""

import argparse
import os
import pickle
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoModelForMaskedLM
from transformers import BertTokenizer
from transformers import DataCollatorForWholeWordMask

from dataset.dataset import StrategizedTokenizerDataset, DefaultTokenizerDataset


def run_loss_benchmark(dataloader, model):
        
    total_loss = 0
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    model.train()
    
    for batch in tqdm(dataloader):
        inputs = {k: v.to(device) for k,v in batch.items()}
        outputs = model.forward(**inputs)
        loss = outputs.loss.item()
        
        del outputs #During local testing it would give memory errors because the outputs arent used in a backward pass
        total_loss += loss*dataloader.batch_size
    
    return total_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True,
                        default="test_experiment/model2/",
                        help='Where the models are')
    parser.add_argument("--model-name", required=True,
                        default="test_experiment/model2/",
                        help='Which pretrained model to finetune')
    parser.add_argument("--cache-dir", required=True,
                        help="Location of pre-made files")
    parser.add_argument("--data-dir", required=True,
                        help="Location of saved pytorch tensors")
    parser.add_argument('--run-mode', required=True,
                        type=str,
                        default='full',
                        help="Whether to run a 1/100 sample or full version of the finetuning.")
    parser.add_argument("--batch_size", required=False,
                        type=int, default=32, 
                        help="Desired batch size")
    parser.add_argument("--dataset", required=True,
                        type=str,
                        default='StrategizedMasking',
                        help='Whether to select the RandomMasking or StrategizedMasking')
    
    
    args = parser.parse_args()
    
    model_dir = args.model_dir
    model_name = args.model_name
    
    cache_dir = args.cache_dir
    data_dir = args.data_dir
    
    dataset = args.dataset
    batch_size = args.batch_size
    
    run_mode = args.run_mode
    
    if run_mode == 'full':
        book_file = 'subset_meta_ratio_100M.pkl'
    elif run_mode == 'test':
        book_file = 'subset_meta_ratio_100K.pkl'
    else:
        raise ValueError('Invalid value for argument --run-mode. Needs to be "full" or "test"')
    
    with open(os.path.join(cache_dir, book_file), 'rb') as f:
        book_list = pickle.load(f)['subset_booklist']
    
    print('Loaded book_list')
    print('Creating dataset object')
    if dataset == 'StrategizedMasking':
        benchmark_dataset = StrategizedTokenizerDataset(datadir=data_dir, max_seq_length=128)
        benchmark_dataset.populate(book_list=book_list)
        dataloader = DataLoader(benchmark_dataset, 
                                batch_size=batch_size)
        
    elif dataset == 'RandomMasking':
        train_dataset_og_bert = DefaultTokenizerDataset(datadir=data_dir, max_seq_length=128)
        train_dataset_og_bert.populate(book_list=book_list)
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, 
                                             mlm=True, 
                                             mlm_probability=0.15)
        dataloader = DataLoader(train_dataset_og_bert, batch_size=batch_size, collate_fn=data_collator)
        
    print('Created dataloader object with populated dataset')
    

    model = AutoModelForMaskedLM.from_pretrained(os.path.join(model_dir, model_name, '0'))

    
    print('Loaded model')
    total_loss = run_loss_benchmark(dataloader, model)
    print('Computing loss complete: {}'.format(total_loss))
    with open(os.path.join(model_dir, model_name, '0', '{}_benchmark_result.pkl'.format(dataset)), 'wb') as f:
        pickle.dump(total_loss, f)
        print('Saved loss to {}'.format(f))
    
    
    

if __name__ == "__main__":
    main()
