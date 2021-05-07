# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 12:03:48 2021

@author: s145733
"""
import pickle
import os
import pandas as pd
import argparse

from pretraining_data_utils import make_book_token_frequency, token_freq_df_to_dict, all_available_tokens_from_df, optimize_book_subset_ratio
          
      
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True,
                        help="Location of pre-made files")
    
    parser.add_argument("--datasizes", default = [1e5, 1e6, 1e7, 1e8], type=list,
                        help="Which sizes of datasets should be created")
    
    args = parser.parse_args()
    
    datasizes = args.datasizes
    cache_dir = args.cache_dir
    
    if os.path.isfile(os.path.join(cache_dir, 'loadable_english_book_keys.pkl')):
        with open(os.path.join(cache_dir, 'loadable_english_book_keys.pkl'), 'rb') as f:
            loadable_english_book_keys = pickle.load(f)
    else:
        raise RuntimeError("loadable_english_book_keys.pkl doesnt exist in provided cache directory. Check notebook on how to make and load or use provided version from github repo")
    
    if os.path.isfile(os.path.join(cache_dir, 'df_book_token_freq.csv')) and \
        os.path.isfile(os.path.join(cache_dir, 'df_total_tokens.csv')):
        print('Loading df_book_token_freq.csv and df_total_tokens.csv from disk')
        df_book_token_freq = pd.read_csv(os.path.join(cache_dir, 'df_book_token_freq.csv'), index_col=0)
        df_total_tokens = pd.read_csv(os.path.join(cache_dir, 'df_total_tokens.csv'), index_col=0).squeeze()
    else:
        print('Creating df_book_token_freq and df_total_tokens from scratch')
        df_book_token_freq, df_total_tokens = make_book_token_frequency(loadable_english_book_keys)
        df_book_token_freq.to_csv(os.path.join(cache_dir, 'df_book_token_freq.csv'))
        df_total_tokens.to_csv(os.path.join(cache_dir, 'df_total_tokens.csv'))
    
        
    all_present_tokens = all_available_tokens_from_df(df_book_token_freq)
    tokens_per_book = token_freq_df_to_dict(df_book_token_freq, df_total_tokens)
    
    for datasize in datasizes:
        print('Optimizing datasize: ', datasize)
        print('Greedy ratio algorithm')
        subset_meta_ratio = optimize_book_subset_ratio(all_present_tokens, tokens_per_book, threshold=datasize)
        print(subset_meta_ratio)
        with open(os.path.join(cache_dir, 'subset_meta_ratio_' + human_format(datasize) + '.pkl'), 'wb') as f:
            pickle.dump(subset_meta_ratio, f)
                

def human_format(num) -> str:
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])
  
  
if __name__ == "__main__":
    main()
    
    