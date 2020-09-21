
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

codeDictionary = {"D":0, "M":1, "S":2, "H":3, "F":4, "O":5, "E":6, "NA":7}

processor = spacy.load('en_core_web_sm')
#global generic
#global brand
global drug
global nondrug

#model_name = "models/" #textattack/bert-base-uncased-QQP"
#model_name = "models/bert-base-uncased.tar.gz" #textattack/bert-base-uncased-QQP"

def load_objects():
    paths = ["drug", "nondrug", "food", "sport" ]
    
    objects = []
    for path in paths:
        with open(path+".txt", "r") as fh:
            objects.append(fh.readlines())

    return objects

objects = load_objects()
drug = objects[0]
nondrug = objects[1]
food = objects[2]
sport = objects[3]


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

"""
generic, brand = clean_drugs()
drug = generic + brand

food = generate_words('I love eating {mask}')
sport = generate_words("I love playing the physical sport called {mask}")
sport += generate_words("I love playing the physical exercise called {mask}")
sport = list(set(sport))
nondrug = generate_words('The doctor prescribed me {mask} and told me to take it after meals')

nondrug = list(set(nondrug) - set(generic) - set(brand))

with open("food.txt", "w") as fh:
    for item in food:
        fh.writelines(item+"\n")
with open("sport.txt", "w") as fh:
    for item in sport:
        fh.writelines(item+"\n")
with open("drug.txt", "w") as fh:
    for item in drug:
        fh.writelines(item+"\n")
with open("nondrug.txt", "w") as fh:
    for item in nondrug:
        fh.writelines(item+"\n")
"""

def swap_nondrug(x, *args, **kwargs):
    # Returns empty or a list of strings with profesions changed
    professions = nondrug #['doctor', 'nurse', 'engineer', 'lawyer']
    ret = []
    for p in professions:
        #print(p, x)
        if re.search(r'\b%s\b' % p, x):
            ret.extend([re.sub(r'\b%s\b' % p, p2, x) for p2 in professions if p != p2])
    return ret

def swap_drug(x, *args, **kwargs):
    # Returns empty or a list of strings with profesions changed
    professions = drug #['doctor', 'nurse', 'engineer', 'lawyer']
    ret = []
    for p in professions:
        if re.search(r'\b%s\b' % p, x):
            ret.extend([re.sub(r'\b%s\b' % p, p2, x) for p2 in professions if p != p2])
    return ret

def swap_dn(x, *args, **kwargs):
    # Returns empty or a list of strings with profesions changed
    professions = drug #['doctor', 'nurse', 'engineer', 'lawyer']
    ret = []
    for p in professions:
        if re.search(r'\b%s\b' % p, x):
            ret.extend([re.sub(r'\b%s\b' % p, p2, x) for p2 in nondrug if p != p2])
    return ret

def swap_nd(x, *args, **kwargs):
    # Returns empty or a list of strings with profesions changed
    professions = nondrug #['doctor', 'nurse', 'engineer', 'lawyer']
    ret = []
    for p in professions:
        if re.search(r'\b%s\b' % p, x):
            ret.extend([re.sub(r'\b%s\b' % p, p2, x) for p2 in drug if p != p2])
    return ret


def object_test():
    """
    codeDictionary = {"D":0, "M":1, "S":2, "H":3, "F":4, "O":5, "E":6, "NA":7}
    """

    editor = Editor()
    food_ret = editor.template('I enjoy having {food}.', food=food, labels=4, save=True) #, nsamples=100)   
    mft_food = MFT(food_ret.data, labels=food_ret.labels, name='Object Rec: Food',
           capability='Objects', description='Food')
    

    sport_ret = editor.template('I enjoy doing {sport}.', sport=sport, labels=6, save=True) #, nsamples=100)   
    mft_sport = MFT(sport_ret.data, labels=sport_ret.labels, name='Object Rec: Sport',
           capability='Objects', description='Sport')

    nondrug_ret = editor.template('I enjoy having {nondrug}.', nondrug=nondrug, labels=5) #, save=True) #, nsamples=100)   
    mft_nondrug = MFT(nondrug_ret.data, labels=nondrug_ret.labels, name='Object Rec: Non Drug',
           capability='Objects', description='Non Drug')

    drug_ret = editor.template('I enjoy having {drug}.', drug=drug, labels=1, save=True) #, nsamples=100)   
    mft_drug = MFT(drug_ret.data, labels=drug_ret.labels, name='Object Rec: Drug',
           capability='Objects', description='Drug')
    

    #print(nondrug_ret.data)


    t = Perturb.perturb(nondrug_ret.data, swap_nondrug)
    inv_n = INV(**t, name='swap nondrug name in both questions', capability='objects',
          description='')


    t = Perturb.perturb(drug_ret.data, swap_drug)
    inv_d = INV(**t, name='swap drug name in both questions', capability='objects',
          description='')

    nondrug_monodec = Expect.monotonic(label=5, increasing=False, tolerance=0.1)
    drug_monodec = Expect.monotonic(label=1, increasing=False, tolerance=0.1)
    
    t = Perturb.perturb(nondrug_ret.data, swap_nd)
    dir_nd = DIR(**t, expect=nondrug_monodec)

    t = Perturb.perturb(drug_ret.data, swap_dn)
    dir_dn = DIR(**t, expect=drug_monodec)

            # diet    #exercise   # other     # medical  # other # medical, # o -> m, # m->o
    tests = [ mft_food, mft_sport, mft_nondrug, mft_drug, inv_n, inv_d, dir_nd, dir_dn ]
    
    return tests

def robustness_test():
    editor = Editor()
    food_ret = editor.template('I enjoy having {food}.', food=food, labels=4, save=True) #, nsamples=100)   
    sport_ret = editor.template('I enjoy doing {sport}.', sport=sport, labels=6, save=True) #, nsamples=100)   
    nondrug_ret = editor.template('I enjoy having {nondrug}.', nondrug=nondrug, labels=5) #, save=True) #, nsamples=100)   
    drug_ret = editor.template('I enjoy having {drug}.', drug=drug, labels=1, save=True) #, nsamples=100)   

    fdata = list(processor.pipe(food_ret.data))
    sdata = list(processor.pipe(sport_ret.data))
    ndata = list(processor.pipe(nondrug_ret.data))
    ddata = list(processor.pipe(drug_ret.data))
    
    # punctuation
    ret = Perturb.perturb(pdata, Perturb.punctuation)

    t = Perturb.perturb(nondrug_ret.data, swap_nondrug)
    inv_n = INV(**t, name='swap nondrug name in both questions', capability='objects',
          description='')
    
    # typo
    ret = Perturb.perturb(data, Perturb.add_typos, nsamples=4, keep_original=True)
    ret.data

    # contraction
    ret = Perturb.perturb(data, Perturb.contractions)
    ret.data
    
    # names, location, number
    #ret = Perturb.perturb(pdata[2:3], Perturb.change_names, nsamples=999, first_only=True, keep_original=False)
    #ret.data

    return

#object_test()
#robustness_test()
#exit()

def main():

    
    
    print(swap_dn("Amaryl is bad for cloud"))
    print(generate_sents('I had {word} last night', food))

    return

    """
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
    """

main()
