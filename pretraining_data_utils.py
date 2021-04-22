# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 10:58:58 2021

"""
from gutenberg.acquire import load_etext
from cleaner_utils import super_cleaner
from transformers import BertTokenizer
import pandas as pd
import logging

def book_properties(book_sentences: list, print_output=False):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    sentence_lengths = [len(x) for x in book_sentences]
    num_sentences = len(book_sentences)
    shortest_sent = min(sentence_lengths)
    longest_sent = max(sentence_lengths)
    
    sentence_inputs = tokenizer(book_sentences,
                            add_special_tokens=False,
                            return_attention_mask=False,
                            return_token_type_ids=False)
    sentence_tokens = [len(x) for x in sentence_inputs['input_ids']]
    longest_sequence = max(sentence_tokens)
    total_tokens = sum(sentence_tokens)
    total_tokens, longest_sequence
    
    if print_output:
        print('Shortest sentence (char): ', shortest_sent)
        print('Longest sentence (char): ', longest_sent)
        print('Total tokens: ', total_tokens)
        print('Longest sequence (tokens): ', longest_sequence)
        
    return [num_sentences, shortest_sent, longest_sent, total_tokens, longest_sequence]


def make_df_book_properties(list_book_ids: list):
    column_names = ['book_id', 'num_sentences', 'Shortest sentence (char)', 'Longest sentence (char)', 'Total tokens', 'Longest sequence (tokens)']
    df = pd.DataFrame(columns = column_names)
    for book_id in list_book_ids:
        if type(book_id) != int:
            book_id = int(book_id)
        try:
            data = book_properties(super_cleaner(load_etext(book_id), -1, verify_deletions=False))            
            df.loc[len(df)] = [book_id] + data
        except:
            df.loc[len(df)] = [book_id] + ['Failed']*5
        
    return df
    
    
    