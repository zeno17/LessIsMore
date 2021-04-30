# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 10:58:58 2021

"""
from gutenberg.acquire import load_etext
from cleaner_utils import super_cleaner
from transformers import BertTokenizer

import logging
import numpy as np
import pandas as pd
import tqdm

import torch
import os

from tokenizer.tokenizer import StrategizedTokenizer


def make_book_token_frequency(book_id_list: list):
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    df = pd.DataFrame(columns=list(tokenizer.vocab))
    i = 0
    for book in tqdm.tqdm(book_id_list):
        text = super_cleaner(load_etext(int(book)), -1, verify_deletions=False, return_list=False)
        inputs = tokenizer(text,
                   return_attention_mask=False, 
                   return_token_type_ids=False,)
        token_freqs = np.bincount(inputs['input_ids'], minlength=len(tokenizer.vocab))

        df.loc[len(df)] = token_freqs
        i += 1
        
    df.index = book_id_list
    df_total_tokens = df[list(df.columns)].sum(axis=1)
    
    return df, df_total_tokens

def token_freq_df_to_dict(df_token_freq, df_total_tokens):
    '''
    make a new dict in the form of e.g.:
    {book_id: {'tokens': [101, 102, 999, ..., 29584], 'total_tokens': 75863}}
    '''
    tokens_per_book = {}
    for index, row in df_token_freq.iterrows():
        tokens_per_book[index] = {'tokens': np.flatnonzero(row) , 'total_tokens': df_total_tokens[index]}
    return tokens_per_book

def all_available_tokens_from_df(df_token_freq):
    return np.flatnonzero(df_token_freq.sum(axis=0))

def optimize_book_subset(all_available_tokens, tokens_per_book, threshold: 1e6):
    subset_total_tokens = 0
    subset_present_tokens = []
    subset_books = []
    
    available_books = list(tokens_per_book.keys())
    
    
    #Ensure we dont go over threshold, check if we already have all available tokens
    while subset_total_tokens <= threshold and len(np.setdiff1d(all_available_tokens, subset_present_tokens)) != 0:
        books = [] #ID's of books
        books_new_tokens = [] #How many new tokens does a specific book introduce
        
        for book_id in available_books:
            if tokens_per_book[book_id]['total_tokens'] >= threshold - subset_total_tokens: #check if adding a book already exceeds values
                available_books.remove(book_id) #remove book
            else:
                books.append(book_id)
                books_new_tokens.append(len(np.setdiff1d(tokens_per_book[book_id]['tokens'], subset_present_tokens, 
                                                         assume_unique=True)))
        
        
        
        if len(books) == 0: #No books left which keep it below the threshold. 
            break
        
        new_num_tokens = np.max(books_new_tokens) #Find maximum number of new possible tokens
        
        if new_num_tokens == 0: #No books left which can introduce new tokens while remaining below threshold
            break
        
        #there may be ties in new amount of tokens
        cand_books = np.array(books)[np.argwhere(books_new_tokens == new_num_tokens).flatten()] 
        
        #Consider which book adds the fewest total amount of tokens
        cand_books_total_tokens = [tokens_per_book[cand_book_id]['total_tokens'] for cand_book_id in cand_books]
        
        #If there are also ties in the total amount of tokens just take the first book
        book_best = cand_books[np.argwhere(cand_books_total_tokens == np.min(cand_books_total_tokens)).flatten()[0]] 
        
        print('book best: ', book_best, 'new tokens: ', len(np.setdiff1d(tokens_per_book[book_best]['tokens'], subset_present_tokens)))
        subset_present_tokens = np.union1d(subset_present_tokens, tokens_per_book[book_best]['tokens'])
        subset_total_tokens += tokens_per_book[book_best]['total_tokens']
        subset_books.append(book_best)
        available_books.remove(book_best)
    
    return {'subset_booklist': subset_books, 
            'subset_total_tokens': subset_total_tokens, 
            'subset_present_tokens': subset_present_tokens,
            'subset_unique_tokens': len(subset_present_tokens)}
                
def optimize_book_subset_ratio(all_available_tokens, tokens_per_book, threshold: 1e6):
    subset_total_tokens = 0
    subset_present_tokens = []
    subset_books = []
    
    available_books = list(tokens_per_book.keys())
    
    
    #Ensure we dont go over threshold, check if we already have all available tokens
    while subset_total_tokens <= threshold and len(np.setdiff1d(all_available_tokens, subset_present_tokens)) != 0:
        books = [] #ID's of books
        books_new_tokens = [] #How many new tokens does a specific book introduce
        books_ratio = []
        
        for book_id in available_books:
            if tokens_per_book[book_id]['total_tokens'] >= threshold - subset_total_tokens: #check if adding a book already exceeds values
                available_books.remove(book_id) #remove book
            else:
                books.append(book_id)
                books_new_tokens.append(len(np.setdiff1d(tokens_per_book[book_id]['tokens'], subset_present_tokens, 
                                                         assume_unique=True)))
                books_ratio.append(np.divide(books_new_tokens[-1], tokens_per_book[book_id]['total_tokens']))
        
        if len(books) == 0: #No books left which keep it below the threshold. 
            break
        
        if max(books_new_tokens) == 0: #No books left which can introduce new tokens while remaining below threshold
            break
        
        best_ratio = np.max(books_ratio) #Find the maximum ratio of new possible tokens
        book_best = np.array(books)[np.argwhere(books_ratio == best_ratio).flatten()][0]
        
        print('book best: ', book_best, 
              'new tokens: ', len(np.setdiff1d(tokens_per_book[book_best]['tokens'], subset_present_tokens)),
              'book_total_tokens: ', tokens_per_book[book_best]['total_tokens'],
              'ratio: ', best_ratio)
        subset_present_tokens = np.union1d(subset_present_tokens, tokens_per_book[book_best]['tokens'])
        subset_total_tokens += tokens_per_book[book_best]['total_tokens']
        subset_books.append(book_best)
        available_books.remove(book_best)
    
    return {'subset_booklist': subset_books, 
            'subset_total_tokens': subset_total_tokens, 
            'subset_present_tokens': subset_present_tokens,
            'subset_unique_tokens': len(subset_present_tokens)}
                            
                


def book_properties(book_sentences, print_output=False):
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


def make_df_book_properties(list_book_ids):
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


class BookWriter(object):
    def __init__(self, datadir, overwrite=True):
        self.datadir = datadir
        self.tokenizer = StrategizedTokenizer()
        self.overwrite = overwrite
        
        if not os.path.exists(self.datadir):
            raise DatadirDoesNotExist('Given datadir does not exist')

    def process_book(self, book_id: int):
        if os.path.exists(path = os.path.join(self.datadir, str(book_id), "tensor_file.pt")):
            if self.overwrite == 'skip':
                return
            elif self.overwrite == False:
                raise TensorFileAlreadyExists('File already exists at {}. Use BookWriter(overwrite=True) if you want to overwrite existing files'.format(os.path.join(self.datadir, str(book_id), "tensor_file.pt")))

        sentences = super_cleaner(load_etext(book_id), -1, verify_deletions=False)
        print('Read book {}. Starting encoding'.format(book_id))
        inputs = self.encode_book(sentences)
        print('Encoded book {}. Starting saving'.format(book_id))
        self.write_encoding_to_file(book_id, inputs)
        
        

    def encode_book(self, sentences):
        inputs = {key: torch.tensor([])  for key in ['input_ids', 'attention_mask', 'labels']}
        for sentence in tqdm(sentences):
            sentence_encoding = self.tokenizer.tokenize(sentence)
            inputs = {key: torch.cat((inputs[key], sentence_encoding[key])) for key,val in inputs.items()}
        
        return inputs
        
    def write_encoding_to_file(self, book_id, inputs):
        directory = os.path.join(self.datadir, str(book_id))
        path = os.path.join(self.datadir, str(book_id), "tensor_file.pt")
        if not os.path.exists(directory):
            os.makedirs(directory)
        elif os.path.exists(directory) and os.path.exists(path):
            if self.overwrite == False:
                raise TensorFileAlreadyExists('File already exists at {}. Use BookWriter(overwrite=True) if you want to overwrite existing files'.format(path))
        torch.save(inputs, path)
        print('Saved book {} to {}.'.format(book_id, path))
        
class Error(Exception):
    """Base class for other exceptions"""
    pass

class TensorFileAlreadyExists(Error):
    """Raised when a tensor_file.pt already exists for a given book_id"""
    pass

class DatadirDoesNotExist(Error):
    """"Raised when the given datadir does not exist"""
    pass




    
    