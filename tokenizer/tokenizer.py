# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 14:26:42 2021
"""
import spacy
import re

from transformers import BertTokenizer

class StrategizedTokenizer(object):
    def __init__(self, pos_based_mask=True, lemmatize=True, ner_based_swap=True):
        """
        Constructs the strategized Tokenizer.
        Loads the required spacy model
        
        Processes the sentence based on desired properties
        
        ==Not guaranteed to work on cased vocabularies==
        """
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.pos_based_mask = pos_based_mask
        self.lemmatize = lemmatize
        self.ner_based_swap = ner_based_swap
        

    def tokenize(self, text):
        spacy_sentence = self.nlp(text, disable=['parser'])
        
        processed_text_list = []
        if self.pos_based_mask:
            processed_text_list += self.mask_text_pos_based(text, spacy_sentence)
        if self.lemmatize:
            processed_text_list += self.lemmatize_text(text, spacy_sentence)         
        if self.ner_based_swap:
            processed_text_list += self.ner_swap_text(text, spacy_sentence)
        #TODO add more?
        
        for x in processed_text_list:
            print(x)
        inputs = self.tokenizer(processed_text_list,
                                return_token_type_ids=False, #Dont need this because we dont use NSP
                                return_tensors="pt", 
                                padding=True)
        inputs['labels'] = self.tokenizer([text for i in range(0,len(processed_text_list))], 
                                          return_attention_mask=False, 
                                          return_token_type_ids=False,
                                          return_tensors='pt', 
                                          padding=True)['input_ids']
        return inputs
    
    def mask_text_pos_based(self, text, spacy_sentence) -> list:
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        pos_masks = []
        
        text_pos_list = [token.pos_ for token in spacy_sentence]
        text_pos_frequency = {pos: text_pos_list.count(pos) for pos in text_pos_list}
        
        for posoi in text_pos_frequency.keys():
            posoi_vocab = [token.text.lower() for token in spacy_sentence if token.pos_ == posoi]
            mask_indices = []
            
            composite_word_indices = []
            composite_word_tokens = []
            for (i, token) in enumerate(input_tokens):
                if token == "[CLS]" or token == "[SEP]":
                    continue
                elif token.startswith("##"):
                    composite_word_indices.append(i)
                    composite_word_tokens.append(token)
                    if "".join([x.strip("##") for x in composite_word_tokens]) in posoi_vocab:
                        mask_indices = mask_indices + composite_word_indices

                elif token in posoi_vocab:
                    mask_indices.append(i)
                else:
                    composite_word_indices = [i]
                    composite_word_tokens = [token]

            mask_labels = [1 if i in mask_indices else 0 for i in range(len(input_tokens))]
            masked_tokens = [x if mask_labels[i] == 0 else 103 for i,x in enumerate(input_ids)]
            masked_input = self.tokenizer.decode(masked_tokens)         
            pos_masks.append(masked_input)
            
        return pos_masks
    
    def lemmatize_text(self, text, spacy_sentence) -> list:
        replacement_tuples = [(token.text, token.lemma_) for token in spacy_sentence if token.text.lower() != token.lemma_]
        lemmatized_text = text
        for replacement in replacement_tuples:
            lemmatized_text = re.sub(r'\b' + replacement[0] + r'\b', replacement[1], lemmatized_text)

        lemmatized_text = lemmatized_text.replace("  ", " ")
        return [lemmatized_text]
    
    def ner_swap_text(self, text, spacy_sentence) -> list:
        ner_swapped_text = spacy_sentence.text
        for ent in spacy_sentence.ents:
            if ent.label_ == 'TIME':
                time_substring = ner_swapped_text[ent.start_char:ent.end_char].split(" ")
                time_substring.reverse()
                ner_swapped_text = ner_swapped_text.replace(ner_swapped_text[ent.start_char:ent.end_char], " ".join(time_substring))
            #TODO add other possible ideas
        return [ner_swapped_text]
        
    def convert_ids_to_tokens(self, input_ids):
        return [self.tokenizer.convert_ids_to_tokens(row) for row in input_ids]