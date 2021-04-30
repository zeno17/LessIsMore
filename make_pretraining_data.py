# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:31:07 2021

"""
from pretraining_data_utils import BookWriter


def main():
    bookwriter = BookWriter(datadir='../pretraining_data', overwrite='skip')
    #bookwriter = BookWriter(datadir='../pretraining_data', overwrite='skip')
    for book in [16968, 1741]:
       bookwriter.process_book(book)
           
if __name__ == "__main__":
    main()