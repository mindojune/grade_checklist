
import numpy as np
import checklist
from checklist.editor import Editor
from checklist.perturb import Perturb
from checklist.test_types import MFT, INV, DIR
from checklist.test_suite import TestSuite
from checklist.expect import Expect

import sys
import spacy


from transformers import pipeline, BertTokenizer, BertForSequenceClassification, BertModel
import torch
import jsonlines

import json
import glob
from collections import Counter
import argparse
import random
import re

processor = spacy.load('en_core_web_sm')
global generic
global brand

#model_name = "models/" #textattack/bert-base-uncased-QQP"
#model_name = "models/bert-base-uncased.tar.gz" #textattack/bert-base-uncased-QQP"

def load_model():
    model_name = "models/7300" #textattack/bert-base-uncased-QQP"


    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    #model = BertForSequenceClassification.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    return model, tokenizer

    """
    editor = Editor()
    ret = editor.template('I like {food}.', food=['lawyer', 'doctor', 'accountant'])
    print(ret.data)
    print(np.random.choice(ret.data, 3))
    """

def jsonl_to_list():
    corpus_root = "./jsonl/"
    path = sorted(glob.glob('{}/*.jsonl'.format(corpus_root)))
    
    data = []
    for file_path in path:

        with jsonlines.open(file_path) as f:
            for line in f.iter():

                utts =  line['sentences'] 
                labels = line['labels']
                #print(utts[0], labels[0])
                data += [ (x,y) for x,y in zip(utts, labels)]
    
    return data


def write_predictions_to_file():

    return

def make_suite():
    suite = TestSuite()
    editor = Editor()
    
    # add tests

def filter_data(data, criterion):
    
    new_d = []
    for datum in data:
        if criterion(datum):
            new_d.append(datum)

    return new_d

def minor(data):
    Perturb.strip_punctuation
    Perturb.punctuation
    Perturb.add_typos
    Perturb.contract
    Perturb.expand_contractions
    Perturb.contractions
    Perturb.change_names
    Perturb.change_location
    Perturb.change_number
    return

def taxonomy(data):
    editor.synonyms('My drink is hot.', 'hot')
    return


def negation(data):
    pdata = list(processor.pipe([x[0] for x in data]))
    print(pdata)

    ret = Perturb.perturb(pdata, Perturb.add_negation)
    return ret

def generate_words(suggest_sentence):
    editor = Editor()
    words = editor.suggest(suggest_sentence)

    return words

def generate_sents(template, words):
    editor = Editor()
    ret = editor.template(template, word=words)
    return ret.data

def clean_drugs():
    generic, brand = [], []
    with open("diabetes_drugs.txt", "r") as fh:
        data = fh.readlines()
        data = [x.strip().split() for x in data]
        
        for datum in data:
            for i in datum:
                i = i.strip("(").strip(")")
                if i[0].isupper():
                    brand.append(i)
                else:
                    generic.append(i)
    return list(set(generic)), list(set(brand))
generic, brand = clean_drugs()

def swap_generic(x, *args, **kwargs):
    # Returns empty or a list of strings with profesions changed
    professions = generic #['doctor', 'nurse', 'engineer', 'lawyer']
    ret = []
    for p in professions:
        if re.search(r'\b%s\b' % p, x):
            ret.extend([re.sub(r'\b%s\b' % p, p2, x) for p2 in professions if p != p2])
    return ret

def swap_brand(x, *args, **kwargs):
    # Returns empty or a list of strings with profesions changed
    professions = brand #['doctor', 'nurse', 'engineer', 'lawyer']
    ret = []
    for p in professions:
        if re.search(r'\b%s\b' % p, x):
            ret.extend([re.sub(r'\b%s\b' % p, p2, x) for p2 in professions if p != p2])
    return ret

def swap_g_to_b(x, *args, **kwargs):
    # Returns empty or a list of strings with profesions changed
    professions = generic #['doctor', 'nurse', 'engineer', 'lawyer']
    ret = []
    for p in professions:
        if re.search(r'\b%s\b' % p, x):
            ret.extend([re.sub(r'\b%s\b' % p, p2, x) for p2 in brand if p != p2])
    return ret

def swap_b_to_g(x, *args, **kwargs):
    # Returns empty or a list of strings with profesions changed
    professions = brand #['doctor', 'nurse', 'engineer', 'lawyer']
    ret = []
    for p in professions:
        if re.search(r'\b%s\b' % p, x):
            ret.extend([re.sub(r'\b%s\b' % p, p2, x) for p2 in generic if p != p2])
    return ret


def voc_pos_ner_test():
    editor = Editor()
    ret = editor.template('This is not {a:pos} {mask}.', pos=pos, labels=0, save=True, nsamples=100)
    ret += editor.template('This is not {a:neg} {mask}.', neg=neg, labels=1, save=True, nsamples=100)
    ret.data

    mft_food = MFT(ret.data, labels=ret.labels, name='Simple negation',
           capability='Negation', description='Very simple negations.')
    
    return mft_food, mft_sport, mft_generic, mft_brand, inv_

def main():

    
    #bert, tokenizer = load_model()
    #inputs = tokenizer("I am a gold collector", return_tensors="pt")
    
    #print(inputs)
    #print(bert(**inputs))
    
    print(swap_b_to_g("Amaryl is bad for cloud"))

    
    
    #print(generic)
    #print(brand)
    #return

    food = generate_words('I love eating {mask}')
    sport = generate_words("I love playing the physical sport called {mask}")
    sport += generate_words("I love playing the physical exercise called {mask}")
    sport = list(set(sport))
    drug = generate_words('The doctor prescribed me {mask} and told me to take it after meals')

    non_drug = set(drug) - set(generic) - set(brand)
    print(non_drug)

    print(len(food))
    print(len(sport))
    print(len(non_drug))


    print(generate_sents('I had {word} last night', food))

    return

    data = jsonl_to_list()
    data = filter_data(data, lambda x: len(x[0]) >= 40 and x[0].strip("Nurse:").strip("Patient:").strip()[0].isupper())
    
    rdx = (np.random.choice(len(data),10, replace=False))
    
    for idx in rdx:
        print(data[idx])
    
    return


    print((data)[3:10])
    
    print(Perturb.perturb(list(processor.pipe(["I am green"])), Perturb.add_negation).data)

    print(negation(data[3:5]))

    return


main()
