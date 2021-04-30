# -*- coding: utf-[8 -*-


from dataset.dataset import MODataset

from transformers import BertConfig
from transformers import BertForMaskedLM
from transformers import Trainer, TrainingArguments

def bert_tiny_config():
    """Config parameters for BERT-tiny"""
    return {"hidden_size": 128, 
            "hidden_act": "gelu", 
            "initializer_range": 0.02, 
            "vocab_size": 30522, 
            "hidden_dropout_prob": 0.1, 
            "num_attention_heads": 2, 
            "type_vocab_size": 2, 
            "max_position_embeddings": 512, 
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
            "max_position_embeddings": 512, 
            "num_hidden_layers": 4, 
            "intermediate_size": 2048, 
            "attention_probs_dropout_prob": 0.1}
    
def bert_base_config():
    """BertConfig class default parameters are for BERT-base so return empty dict."""
    return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                        help="Location of saved pytorch tensors")
    parser.add_argument("--cache-dir", required=True,
                        help="Location of pre-made files")
    parser.add_argument("--dataset", required=True,
                        help="Which datasize to run on")
    
    
    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    cache_dir = args.cache_dir
    
    


if __name__ == "__main__":
    
    
    main()
    train_dataset = MODataset(datadir='../pretraining_data')
    train_dataset.populate()
    


    model = BertForMaskedLM(config=BertConfig(**bert_tiny_config))
    model.train();
    
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total # of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
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
    
    print("Everything ready for training")
    
    trainer.train()
    print("Finished training")
    
    