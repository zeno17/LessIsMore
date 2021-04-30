# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 12:03:48 2021

@author: s145733
"""
import pickle
import os
import pandas as pd

from pretraining_data_utils import make_book_token_frequency, token_freq_df_to_dict, all_available_tokens_from_df, optimize_book_subset, optimize_book_subset_ratio

def main():  
    datasizes = [1e5, 1e6, 1e7]
    if os.path.isfile('cached_files/loadable_english_book_keys.pkl'):
        with open('cached_files/loadable_english_book_keys.pkl', 'rb') as f:
            english_loadable_book_keys = pickle.load(f)
    else:
        raise NoLoadableBookFile()
    
    if os.path.isfile('cached_files/df_book_token_freq.csv') and os.path.isfile('cached_files/df_total_tokens.csv'):
        print('Loading df_book_token_freq.csv and df_total_tokens.csv from disk')
        df_book_token_freq = pd.read_csv('cached_files/df_book_token_freq.csv', index_col=0)
        df_total_tokens = pd.read_csv('cached_files/df_total_tokens.csv', index_col=0).squeeze()
    else:
        print('Creating df_book_token_freq and df_total_tokens from scratch')
        df_book_token_freq, df_total_tokens = make_book_token_frequency(english_loadable_book_keys)
        df_book_token_freq.to_csv('cached_files/df_book_token_freq.csv')
        df_total_tokens.to_csv('cached_files/df_total_tokens.csv')
    
        
    all_present_tokens = all_available_tokens_from_df(df_book_token_freq)
    tokens_per_book = token_freq_df_to_dict(df_book_token_freq, df_total_tokens)
    
    for datasize in datasizes:
        print('Optimizing datasize: ', datasize)
        print('Greedy algorithm')
        subset_meta = optimize_book_subset(all_present_tokens, tokens_per_book, threshold=datasize)
        print(subset_meta)
        print('Greedy ratio algorithm')
        subset_meta_ratio = optimize_book_subset_ratio(all_present_tokens, tokens_per_book, threshold=datasize)
        print(subset_meta_ratio)
        with open('cached_files/subset_meta_' + human_format(datasize) + '.pkl', 'wb') as f:
            pickle.dump(subset_meta, f)
        with open('cached_files/subset_meta_ratio_' + human_format(datasize) + '.pkl', 'wb') as f:
            pickle.dump(subset_meta_ratio, f)
                
            
            
            
def human_format(num) -> str:
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])
            
    
      
      
class Error(Exception):
    """Base class for other exceptions"""
    pass

class NoLoadableBookFile(Error):
    """loadable_english_book_keys.pkl doesnt exist. Check notebook on how to make and load or use provided version from github repo"""
    pass      
  
  
if __name__ == "__main__":
    main()
    
    