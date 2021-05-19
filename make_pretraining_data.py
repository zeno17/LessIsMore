# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:31:07 2021

"""
from pretraining_data_utils import SentenceWriter
from pretraining_data_utils import TensorWriter

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
    parser.add_argument("--book_id_file", required=True,
                        help="Which file with book_ids to use. e.g. 'loadable_english_book_keys.pkl' or  'subset_meta_ratio_100K'")
    parser.add_argument("--truncation", required=True,
                        help='Whether or not to truncate sentences')
    parser.add_argument("--split-sizes", type=str,
                        help="What sizes to split the data in if neccessary")
    parser.add_argument("--pos-based-mask", type=bool,
                        default=True,
                        help='Whether to add the POS-based mask to sentence writer')
    parser.add_argument("--lemmatize", type=bool,
                        default=True,
                        help='Whether to add the lemmatizer mask to sentence writer')
    parser.add_argument("--ner-based-swap", type=bool,
                        default=True,
                        help='Whether to add the ner-based swap mask to sentence writer')

    
    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    cache_dir = args.cache_dir
    book_id_file = args.book_id_file
    truncation = args.truncation

    pos_based_mask = args.pos_based_mask
    lemmatize = args.lemmatize
    ner_based_swap = args.ner_based_swap

    split_sizes = [int(x) for x in args.split_sizes.strip('[').strip(']').split(',')]
        
    sentencewriter = SentenceWriter(datadir=data_dir, truncate=truncation, split_sizes=split_sizes)
    
    tensorwriter = TensorWriter(datadir=data_dir, 
                                split_sizes=split_sizes, 
                                pos_based_mask=pos_based_mask, 
                                lemmatize=lemmatize, 
                                ner_based_swap=ner_based_swap)
    
    #Retrieve book list
    if os.path.isfile(os.path.join(cache_dir, book_id_file)):
        with open(os.path.join(cache_dir, book_id_file), 'rb') as f:
            subset_book_file = pickle.load(f)
            
            if book_id_file.startswith('subset_meta_ratio_'):
                book_ids_list = subset_book_file['subset_booklist']
            else:
                book_ids_list = subset_book_file
    else:
        raise RuntimeError("Provided book_id_file doesnt exist in provided cache directory.")
    print(truncation, split_sizes, book_ids_list)
    
    for book_id in tqdm(book_ids_list):
        sentencewriter.process_book(book_id)
      
    print('Read all books')
    for book_id in tqdm(book_ids_list):
        tensorwriter.encode_book(book_id)
    print('Wrote all books into tensors')    
           
if __name__ == "__main__":
    main()