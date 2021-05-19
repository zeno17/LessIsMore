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
from tqdm import tqdm

import torch
import os
import pickle
import spacy

from tokenizer.tokenizer import StrategizedTokenizer
from pathlib import Path


def make_book_token_frequency(book_id_list: list):
    '''
    Takes in a list of book_ids which are valid ids for gutenberg.org, cleans every book using the supercleaner, and reports back how many tokens there will be remaining
    '''
    
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    df = pd.DataFrame(columns=list(tokenizer.vocab))
    i = 0
    for book in tqdm(book_id_list):
        text = super_cleaner(load_etext(int(book)), -1, verify_deletions=False, return_list=False)
        inputs = tokenizer(text,
                           add_special_tokens=False,
                           return_attention_mask=False, 
                           return_token_type_ids=False)
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
            if tokens_per_book[book_id]['total_tokens'] >= threshold - subset_total_tokens or tokens_per_book[book_id]['total_tokens'] == 0: #check if adding a book already exceeds values
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


class SentenceWriter(object):
    '''
    Reads PG books and places them into pickle files
    '''
    def __init__(self, datadir, truncate, split_sizes=[8,32,128]):
        self.datadir = datadir
        self.truncate = truncate
        self.split_sizes = split_sizes
        
        if not os.path.exists(self.datadir):
            raise DatadirDoesNotExist('Given datadir does not exist')

    def process_book(self, book_id):
        split_sentences = self.make_data_splits(int(book_id), max_seq_lengths=self.split_sizes, truncate=self.truncate)
        for length, sentences in split_sentences.items():
            Path(self.datadir, str(book_id)).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(self.datadir, str(book_id), 'sentences_' + str(length) + '.pkl'), 'wb') as f:
                pickle.dump(sentences, f)
            
    def make_data_splits(self, book_id, max_seq_lengths: list, truncate='chunk') -> dict:
        '''
        Splits the data based on sequence length and places it in a dictionary
        
        Example output:
        {8: ['I am happy', 'That is good'],
        32: ['That is definitely a good thing but couldn't he have told us earlier? Now we are late once again.],
        128: ['<some very long sequence>']}
        '''
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        text_splits = {key: [] for key in max_seq_lengths}
        
        max_position_length = max(max_seq_lengths)
        
        paragraphs = np.array(super_cleaner(load_etext(int(book_id)), -1))
        
        if len(paragraphs) == 0:
            return
        paragraphs_tokens = tokenizer(paragraphs.tolist(),
                                      add_special_tokens=True,
                                      truncation=False,
                                      return_tensors='np')['input_ids']
        max_seq_lengths = [0] + max_seq_lengths
        
        length_tuples = [(max_seq_lengths[i], max_seq_lengths[i+1]) for i in range(0, len(max_seq_lengths)-1)]
        for length_tuple in length_tuples:
            indices = np.flatnonzero((length_tuple[0] < np.array([len(tokens) for tokens in paragraphs_tokens])) & \
                                     (np.array([len(tokens) for tokens in paragraphs_tokens]) <= length_tuple[1]))
                                             
            text_splits[length_tuple[1]] += paragraphs[indices].tolist()
        
        idx_too_long = np.flatnonzero(np.array([len(tokens) for tokens in paragraphs_tokens]) > max_position_length)
        if truncate == True:    
            #Just add everything which is too long to the maximum length for later truncation.
            text_splits[max_position_length] += paragraphs[idx_too_long].tolist()
        
        elif truncate == 'chunk':
            SC = SentenceChunker()
            #Turn the sentences into chunks using SentenceChunker
            for i, sentence in enumerate(paragraphs[idx_too_long]):    
                new_sents = SC.sentence_chunker(str(sentence), max_position_length)        
                text_splits[max_position_length] += new_sents   
                
        return text_splits
            
            
class SentenceChunker(object):
    '''
    Custom class designed to chunk sequences of a 'too_long' length, into multiple chunks which are just longer.
    e.g. a sequence of 300 tokens may get chunked into a pair of sequences with length 170 and 128.
    It corrects for adding the special tokens.
    '''
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    
    def sentence_chunker(self, paragraph, max_seq_length=128, return_tokens=False, return_strings=True):
        doc = self.nlp(paragraph.strip())
        split_paragraph = [sent.text for sent in doc.sents]
        tokenized_sentences = self.tokenizer(split_paragraph,
                                             add_special_tokens=False)['input_ids']
        combined_tokens = [[]]
        combined_sentences = ['']
        for i, tokens in enumerate(tokenized_sentences):
            if len(combined_tokens[-1]) + len(tokens) <= (max_seq_length - 2):
                combined_tokens[-1] += tokens
                if combined_sentences[-1].endswith('.') or combined_sentences[-1].endswith('?') or combined_sentences[-1].endswith('!'): #prevent stuff being concatenated on a dot
                    combined_sentences[-1] += ' ' + split_paragraph[i]
                else:
                    combined_sentences[-1] += split_paragraph[i]
            else:
                combined_tokens[-1] += tokens
                combined_sentences[-1] += split_paragraph[i]
                if combined_sentences[-1].startswith('" '):
                    combined_sentences[-1] = combined_sentences[-1].lstrip('" ')
                    combined_tokens[-1] = combined_tokens[-1][1:]
                combined_tokens.append([])
                combined_sentences.append('')
                
        if return_tokens and return_strings:
            return combined_tokens, combined_sentences
        elif return_tokens:
            return combined_tokens
        elif return_strings:
            return combined_sentences
        
                
class TensorWriter(object):
    def __init__(self, datadir, split_sizes, pos_based_mask, lemmatize, ner_based_swap):
        self.datadir = datadir
        self.split_sizes = split_sizes
        self.tokenizers = {split_size: StrategizedTokenizer(padding='max_length', 
                                                            truncation=True, 
                                                            max_seq_length=split_size,
                                                            pos_based_mask=pos_based_mask, 
                                                            lemmatize=lemmatize, 
                                                            ner_based_swap=ner_based_swap) for split_size in split_sizes}
    
    
    def encode_book(self, book_id):
        directory = os.path.join(self.datadir, str(book_id))
        files = [x for x in os.listdir(directory) if x.endswith('.pkl')]
        
        for file in files:
            with open(os.path.join(directory, file), 'rb') as f:
                sentences = pickle.load(f)
            
            max_seq_length = int(file.strip('.pkl').split('_')[-1])
            inputs = {key: torch.tensor([])  for key in ['input_ids', 'attention_mask', 'labels']}
            for sentence in sentences:
                #select the right tokenizer
                sentence_encoding = self.tokenizers[max_seq_length].tokenize(sentence)
                inputs = {key: torch.cat((inputs[key], sentence_encoding[key])) for key,val in inputs.items()}
        
            torch.save(inputs, os.path.join(directory, 'tensors_' + str(max_seq_length) + '.pt'))
            print('Saved encodings of book {} to {}.'.format(book_id, os.path.join(directory, 'tensors_' + str(max_seq_length) + '.pt')))

        
class Error(Exception):
    """Base class for other exceptions"""
    pass

class DatadirDoesNotExist(Error):
    """"Raised when the given datadir does not exist"""
    pass




    
    