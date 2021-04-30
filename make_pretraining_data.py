# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:31:07 2021

"""
from pretraining_data_utils import BookWriter

from tqdm import tqdm
import argparse
import os
import pickle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                        help="Location of saved pytorch tensors")
    parser.add_argument("--cache-dir", required=True,
                        help="Location of pre-made files")
    
    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    cache_dir = args.cache_dir
    bookwriter = BookWriter(datadir=data_dir, overwrite='skip')
    

    #Retrieve book list
    if os.path.isfile(os.path.join(cache_dir, 'loadable_english_book_keys.pkl')):
        with open(os.path.join(cache_dir, 'loadable_english_book_keys.pkl'), 'rb') as f:
            loadable_english_book_keys = pickle.load(f)
    else:
        raise RuntimeError("loadable_english_book_keys.pkl doesnt exist in provided cache directory. Check notebook on how to make and load or use provided version from github repo")
    
    
    for book in tqdm(loadable_english_book_keys[:10]):
       bookwriter.process_book(book)
           
if __name__ == "__main__":
    main()