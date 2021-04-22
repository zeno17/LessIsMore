# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:33:39 2021

"""
import spacy
from datetime import datetime
import re
import torch
    
    
def whole_word_MO_tokenization_and_masking(tokenizer, nlp_model, sequence: str):
        """
        posoi: Part-Of-Speech of interest
        
        Performs whole-word-masking based on selected posoi.
        
        POS possibilities:['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 
                            'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
                             
        TODO: What if no tokens are masked?
        
        """
        print('loading:', datetime.now().time())
        spacy_sentence = nlp_model(sequence, disable=["parser"])
        
        POS_list = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 
                            'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
        NER_list = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 
                    'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
        NER_pairs = ['']
        
        input_ids = tokenizer.encode(sequence, add_special_tokens=False)
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids)

        sequence_pos_list = [token.pos_ for token in spacy_sentence]
        sequence_pos_frequency = {pos: sequence_pos_list.count(pos) for pos in sequence_pos_list}
        
        modified_input_list = []
        
        #POS-based masking
        for posoi in sequence_pos_frequency.keys():
            posoi_vocab = [token.text for token in spacy_sentence if token.pos_ == posoi]
            
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
            masked_input = tokenizer.decode(masked_tokens)         
            modified_input_list.append(masked_input)
            
            
        
        #POS-based lemmatization
        replacement_tuples = [(token.text, token.lemma_) for token in spacy_sentence if token.text.lower() != token.lemma_]
        pos_replaced_sentence = sequence
        for replacement in replacement_tuples:
            pos_replaced_sentence = re.sub(r'\b' + replacement[0] + r'\b', replacement[1], pos_replaced_sentence)

        pos_replaced_sentence = pos_replaced_sentence.replace("  ", " ")
        modified_input_list.append(pos_replaced_sentence)
        
        
        #NER-based swapping of time-place (if present)
        ner_swapped_sentence = spacy_sentence.text
        for ent in spacy_sentence.ents:
            if ent.label_ == 'TIME':
                time_substring = ner_swapped_sentence[ent.start_char:ent.end_char].split(" ")
                time_substring.reverse()
                ner_swapped_sentence = ner_swapped_sentence.replace(ner_swapped_sentence[ent.start_char:ent.end_char], " ".join(time_substring))
        #print('NER', ner_swapped_sentence)
        modified_input_list.append(ner_swapped_sentence)
        
        
        #TODO future ideas
        #
        #
        
        inputs = tokenizer(modified_input_list, return_tensors="pt", padding='max_length')

        inputs['labels'] = tokenizer([sequence for i in range(0,inputs['input_ids'].shape[0])], 
                                     return_attention_mask=False, 
                                     return_token_type_ids=False,
                                     return_tensors='pt', padding='max_length')['input_ids']
        return inputs
    
    
class MODataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = {key: val for key, val in encodings.items() if key != 'labels'}
        self.labels = encodings['labels']
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
    
    def __len__(self):
        return len(self.labels)
