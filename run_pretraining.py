# -*- coding: utf-[8 -*-


from dataset.dataset import StrategizedTokenizerDataset

from transformers import BertConfig
from transformers import BertForMaskedLM
from transformers import Trainer, TrainingArguments

import argparse

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
    """BertConfig class default parameters are for BERT-base so return empty dict."""
    return {"max_position_embeddings": 128}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    
    
    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    cache_dir = args.cache_dir
    book_set = args.book_set
    train_batch_size = args.train_batch_size
    config = args.model_config
    
    if config == 'bert-tiny':
        model_config = bert_tiny_config()
    elif config == 'bert-small':
        model_config = bert_small_config()
    elif config == 'bert-base':
        model_config = bert_base_config()
    
    
    if book_set == 'local_test':
        book_list = [16968]
    elif book_set == 'hpc':
        book_list = [10,11,12,13]
    else:
        book_list = []
    
    train_dataset = StrategizedTokenizerDataset(datadir=data_dir)
    train_dataset.populate(book_list=book_list)
    
    model = BertForMaskedLM(config=BertConfig(**model_config))
    model.train();

    training_args = TrainingArguments(
        output_dir='results',
        overwrite_output_dir = True,
        #num_train_epochs=3,              # total # of training epochs
        max_steps = 1,
        warmup_ratio=0.1,
        per_device_train_batch_size=train_batch_size,  # batch size per device during training
        learning_rate=1e-5,     
        logging_dir='logs'             # directory for storing logs
    )
    
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=None            # evaluation dataset
    )
    
    trainer.train()
    trainer.save_model('results')


if __name__ == "__main__":
    main()
    
    