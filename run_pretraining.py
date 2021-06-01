# -*- coding: utf-[8 -*-


from dataset.dataset import StrategizedTokenizerDataset
from dataset.dataset import DefaultTokenizerDataset

from transformers import BertConfig
from transformers import BertForMaskedLM
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorForWholeWordMask


import argparse
import os
import pickle

def bert_tiny_config():
    """Config parameters for BERT-tiny"""
    return {"hidden_size": 128, 
            "hidden_act": "gelu", 
            "initializer_range": 0.02, 
            "vocab_size": 30522, 
            "hidden_dropout_prob": 0.1, 
            "num_attention_heads": 2, 
            "type_vocab_size": 2, 
            "max_position_embeddings": 128, 
            "num_hidden_layers": 2, 
            "intermediate_size": 512, 
            "attention_probs_dropout_prob": 0.1}

def bert_small_config():
    """Config parameters for BERT-small"""
    return {"hidden_size": 512, 
            "hidden_act": "gelu", 
            "initializer_range": 0.02, 
            "vocab_size": 30522, 
            "hidden_dropout_prob": 0.1, 
            "num_attention_heads": 8, 
            "type_vocab_size": 2, 
            "max_position_embeddings": 128, 
            "num_hidden_layers": 4, 
            "intermediate_size": 2048, 
            "attention_probs_dropout_prob": 0.1}
    
def bert_base_config():
    """BertConfig class default parameters are for BERT-base so empty dict except for the things we change."""
    return {"max_position_embeddings": 128}


def train_model(args, trial):
    output_dir = args.output_dir
    data_dir = args.data_dir
    cache_dir = args.cache_dir
    book_set = args.book_set
    train_batch_size = args.train_batch_size
    config = args.model_config
    training_method = args.training_method
    steps_distribution = args.steps_distribution
    print(steps_distribution)
    if isinstance(steps_distribution, str):
        steps_distribution = [int(x) for x in args.steps_distribution.strip('[').strip(']').split(',')]
    
    ####
    # Build model
    ####
    if config == 'bert-tiny':
        model_config = bert_tiny_config()
    elif config == 'bert-small':
        model_config = bert_small_config()
    elif config == 'bert-base':
        model_config = bert_base_config()
    
    model = BertForMaskedLM(config=BertConfig(**model_config))
    model.train();
    
    
    ####
    # Obtain which books to use
    ####
    with open(os.path.join(cache_dir, book_set), 'rb') as f:
        book_file = pickle.load(f)
    
    if book_set.startswith('subset_meta_ratio_'):
        book_list = book_file['subset_booklist']
    else:
        book_list = book_file
    
    
    model_dir = os.path.join(output_dir, str(trial))
        
    if training_method == 'standard':
        split_sizes = [8, 32, 128]
        steps_distribution = steps_distribution
        batch_size_multiplier = [16, 4, 1]
        
        
        for i, length in enumerate(split_sizes):
            train_dataset = StrategizedTokenizerDataset(datadir=data_dir, max_seq_length=length)
            train_dataset.populate(book_list=book_list)
            
            training_args = TrainingArguments(
                output_dir=model_dir,
                overwrite_output_dir=True,
                save_strategy='no',
                
                #Training params we picked because of training strategy
                max_steps = steps_distribution[i],
                per_device_train_batch_size=train_batch_size*batch_size_multiplier[i],  
                #num_train_epochs=3 # For if you want run epochs instead of steps
                
                #Hyper parameters as per BERT-paper which are not default values in TrainingArguments
                warmup_ratio=0.1,
                learning_rate=1e-4,
                weight_decay=0.01
            )
            
            trainer = Trainer(
                model=model,                         # the instantiated 🤗 Transformers model to be trained
                args=training_args,                  # training arguments, defined above
                train_dataset=train_dataset,         # training dataset
                eval_dataset=None                    # evaluation dataset
            )
            
            
            trainer.train()
        
        trainer.save_model(model_dir)
    elif training_method == 'single_length':
        split_sizes = [128]
        steps_distribution = [sum(steps_distribution)]
        batch_size_multiplier = [1]
        
        for i, length in enumerate(split_sizes):
            train_dataset = StrategizedTokenizerDataset(datadir=data_dir, max_seq_length=length)
            train_dataset.populate(book_list=book_list)
            
            training_args = TrainingArguments(
                output_dir=model_dir,
                overwrite_output_dir=True,
                save_strategy='no',
                
                #Training params we picked because of training strategy
                max_steps = steps_distribution[i],
                per_device_train_batch_size=train_batch_size*batch_size_multiplier[i],  
                #num_train_epochs=3 # For if you want run epochs instead of steps
                
                #Hyper parameters as per BERT-paper which are not default values in TrainingArguments
                warmup_ratio=0.1,
                learning_rate=1e-4,
                weight_decay=0.01
            )
            
            trainer = Trainer(
                model=model,                         # the instantiated 🤗 Transformers model to be trained
                args=training_args,                  # training arguments, defined above
                train_dataset=train_dataset,         # training dataset
                eval_dataset=None                    # evaluation dataset
            )
            
            
            trainer.train()
        
        trainer.save_model(model_dir)
        
    elif training_method == 'original_bert':
        split_sizes = [8, 32, 128]
        steps_distribution = steps_distribution
        batch_size_multiplier = [16, 4, 1]
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, 
                                                                     mlm=True, 
                                                                     mlm_probability=0.15)
        for i, length in enumerate(split_sizes):
            train_dataset = DefaultTokenizerDataset(datadir=data_dir, max_seq_length=length)
            train_dataset.populate(book_list=book_list)        
        
            training_args = TrainingArguments(
                output_dir=model_dir,
                overwrite_output_dir=True,
                save_strategy='no',
                
                #Training params we picked because of training strategy
                max_steps = steps_distribution[i],
                per_device_train_batch_size=train_batch_size*batch_size_multiplier[i],  
                #num_train_epochs=3 # For if you want run epochs instead of steps
                
                #Hyper parameters as per BERT-paper which are not default values in TrainingArguments
                warmup_ratio=0.1,
                learning_rate=1e-4,
                weight_decay=0.01
            )
            
            trainer = Trainer(
                model=model,                         # the instantiated 🤗 Transformers model to be trained
                args=training_args,                  # training arguments, defined above
                tokenizer=tokenizer,
                data_collator=data_collator,
                train_dataset=train_dataset,         # training dataset
                eval_dataset=None                    # evaluation dataset
            )
            
            trainer.train()
        
        trainer.save_model(model_dir)
    
    
    

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True,
                        help="Where to save trained models and metadata about training process")
    parser.add_argument("--data-dir", required=True,
                        help="Location of saved pytorch tensors")
    parser.add_argument("--cache-dir", required=True,
                        help="Location of pre-made files")
    parser.add_argument("--book_set", required=False,
                        type=str,
                        help="Which datasize to run on")
    parser.add_argument("--train_batch_size", required=False,
                        type=int, default=256, 
                        help="Desired batch size")
    parser.add_argument("--model-config", required=True,
                        help='Which version of BERT to train on.')
    parser.add_argument('--training-method', required=True,
                        type=str,
                        help='Which training regime to enforce')
    parser.add_argument('--sample-size', required=True,
                        default=1,
                        type=int,
                        help='How many runs to perform')
    parser.add_argument('--steps-distribution', required=False,
                         default=[30000, 30000, 40000],
                         type=str,
                         help='How to split steps across 3 different sizes (8/32/128) or how many steps for single length. Coerces with standard')
    
    args = parser.parse_args()
    
    
    sample_size = args.sample_size
    
    for trial in range(0, sample_size):
        train_model(args, trial)
        


if __name__ == "__main__":
    main()
    
    