import pympi
import glob     # Import glob to easily loop over files
import sys
import json
import csv
import random
from collections import Counter
#import torch
#from transformers import *
#import spacy
import os
import argparse
import numpy as np
from sklearn import model_selection

import nltk.tokenize as tokenize
import nltk.tokenize.texttiling as tiling
from unidecode import unidecode
import re 
from string import punctuation
from text_segment import depth_text_seg, dt_wrapper
import spacy
from spacy.lang.en import English
import gensim
import argparse
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import json
import pickle
from gensim import corpora



MAX = 256

codeDictionary = {"D":0, "M":1, "S":2, "H":3, "F":4, "O":5, "E":6, "NA":7}
five = {"D":0, "M":1, "S":2, "H":2, "F":3, "O":3, "E":0, "NA":4}
three = {"D":0, "M":0, "S":0, "H":0, "F":1, "O":1, "E":0, "NA":2}
two = {"D":0, "M":0, "S":0, "H":0, "F":0, "O":0, "E":0, "NA":1}


def parse_args():
    """Parse command line arguments."""

    #eaf_path = "../data/annotations/"
    #json_path = "../data/json_data/"

    parser = argparse.ArgumentParser(description="EAF parsing and dataset creation script")
    parser.add_argument("--eaf_path", default="../data/annotations/", type=str,
                        help="input eaf folder path.")
    parser.add_argument("--json_path", default="../data/json_data/", type=str,
                        help="output json folder path.")
    parser.add_argument("--print_stats", default=False,  \
            action = 'store_true',  help="Print data statistics.")
    parser.add_argument("--tsv_path", default="../data/tsv_data/", type=str,
                        help="output tsv folder path.")
    parser.add_argument("--chunk_length", default=128, type=int,
                        help="Chunk length. Choose from 128, 256, 512")
    parser.add_argument("--cuda", default=False, \
                       action = 'store_true',  help="Use CUDA.")
    parser.add_argument("--length", default=10, type=int,
                        help="SSC sentence length to use.")
    parser.add_argument("--level", default=8, type=int,
                        help="Code system to use.")
    parser.add_argument("--k", default=5, type=int,
                        help="k to use in k-fold validation.")
    parser.add_argument("--prep_gen", default=False, \
                       action = 'store_true',  help="Prepare for data generation by creating corpus.")
    parser.add_argument("--skip", default=False, \
                       action = 'store_true',  help="Skip the data generation part.")
    parser.add_argument("--lda", default=0, type=int, \
                        help="Pick num topics to Generate LDA pretraining data.")




    return parser.parse_args()

args = parse_args()   
level = args.level
length = args.length
k = args.k
pg = args.prep_gen
skip = args.skip
lda = args.lda

if level == 2:
    dictionary = two
elif level == 3:
    dictionary = three
elif level == 5:
    dictionary = five
elif level == 8:
    dictionary = codeDictionary
 

def getTopic(matches,start):
    topic="NA"
    for match in matches:
        #print ("MATCHES",match[1])
        if match[0] ==start:
            if (not match[2]) or "na" in match[2].lower():
                 topic="NA"
                 #main(sys.argv[1])
            else:
                topic = match[2]
    return topic

def writeToJSON(corpus_root, json_path):
    topics_tiers = ['gradeTopics1', 'gradeTopics2']
    speaker_tiers=['nurse','patient']
    
    count = 0

    topics_counts = {'D':0,'M':0,'S':0,'H':0,'F':0,'O':0,'E':0,'NA':0}
    nurse_counts = {'D':0,'M':0,'S':0,'H':0,'F':0,'O':0,'E':0,'NA':0}
    patient_counts = {'D':0,'M':0,'S':0,'H':0,'F':0,'O':0,'E':0,'NA':0}

    # Loop over all elan files the corpusroot subdirectory called eaf
    
    print("corpus path: ", corpus_root)
    for file_path in glob.glob('{}/*.eaf'.format(corpus_root)):
    # Initialize the elan file
        count += 1
        
        print (file_path)
        eaf= pympi.Elan.Eaf(file_path)

        merged = "merged"
        eaf.add_tier(merged, ling='default-lt', parent=None, locale=None, part=None, ann=None, language=None, tier_dict=None)


        for speaker_tier in speaker_tiers:
            if speaker_tier not in eaf.get_tier_names():
                print ('WARNING!!!')
                print ('One of the speaker tiers is not present in the elan file')
                print ('namely: {}. skipping this one...'.format(speaker_tier))

            # If the tier is present we can loop through the annotation data
            else:
                for utterance in eaf.get_annotation_data_for_tier(speaker_tier):
                    
                    if not utterance[2]:
                        continue
                    matches = eaf.get_annotation_data_at_time('gradeTopics1',utterance[0])
                    
                    topic = getTopic(matches, utterance[0])

                    if speaker_tier == "nurse":
                        new = "N::"+utterance[2]
                    elif speaker_tier == "patient":
                        new = "P::"+utterance[2]
                    
                    new = new+"::"+topic
                    
                    start = utterance[0]
                    end = utterance[1]

                    if start < end:
                        eaf.add_annotation(merged, start, end, value=new, svg_ref=None)
                    else:
                        #print("some funky situation here, probably annotation error")
                        eaf.add_annotation(merged, start, start+1, value=new, svg_ref=None)
        

        annotations = eaf.get_annotation_data_for_tier(merged)
        annotations = sorted(annotations, key=lambda tup: tup[0])
        
        json_file_path = json_path+file_path.split("/")[-1][:-4]+".json"
        
        data = {}
        
        index = 0

        nlp = spacy.load('en_core_web_sm')
        
        for utterance in annotations:     
            vals = utterance[2].split("::")

            tokens = nlp(vals[1])
            #item["content"] = tokens
            
            if vals[0] == "N":
                pre = "Nurse: "
            elif vals[0] == "P":
                pre = "Patient: "
            else:
                print("error!")
                exit()
            separated = tokens.text.split(".")
            
            for sentence in separated:
                if sentence.strip() == "":
                    continue
                data[index] = {"sessionID":count, "start": utterance[0], "end":utterance[1], "content": pre+sentence, "speaker":vals[0], "code":vals[2], "file":file_path }
                index += 1
            

        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent = 4)
          

    return

def getStatistics(data):
    # prints utterance statistics
    #   - average sentence length for nurse / patient utterance
    #   - average sentence length per code for nurse / patient
    #   - distribution of codes for nurse / patients
    #   - record per session-level info too
    # use counter
    
    
    max_sentence_length = -float("inf")
    # average sentence length
    # fraction of sentences
    sessions = {}
    for item in data:
        n = len(item["content"].split()) #not accuratea
        #n = len(item["content"]) 
        if n == 159:
            print(item)
        max_sentence_length = max(max_sentence_length, n)
        
        
        if item["sessionID"] in sessions:
            sessions[item["sessionID"]][item["speaker"]][item["code"]] += 1
            sessions[item["sessionID"]]["N"+item["speaker"]][item["code"]] += n
        else:
            sessions[item["sessionID"]] = {"N":{'D':0,'M':0,'S':0,'H':0,'F':0,'O':0,'E':0,'NA':0}, \
                    "P":{'D':0,'M':0,'S':0,'H':0,'F':0,'O':0,'E':0,'NA':0}, \
                     "NP":{'D':0,'M':0,'S':0,'H':0,'F':0,'O':0,'E':0,'NA':0}, \
                     "NN":{'D':0,'M':0,'S':0,'H':0,'F':0,'O':0,'E':0,'NA':0}
                    }
            sessions[item["sessionID"]][item["speaker"]][item["code"]] += 1
            sessions[item["sessionID"]]["N"+item["speaker"]][item["code"]] += n

    
    N = Counter({'D':0,'M':0,'S':0,'H':0,'F':0,'O':0,'E':0,'NA':0})
    P = Counter({'D':0,'M':0,'S':0,'H':0,'F':0,'O':0,'E':0,'NA':0})
    NN = Counter({'D':0,'M':0,'S':0,'H':0,'F':0,'O':0,'E':0,'NA':0})
    NP = Counter({'D':0,'M':0,'S':0,'H':0,'F':0,'O':0,'E':0,'NA':0})
    
    max_word_count = -float("inf")

    for session in sessions:       
        #print(session, sessions[session])
        N = N + Counter(sessions[session]["N"])
        P = P + Counter(sessions[session]["P"])
        NN = NN + Counter(sessions[session]["NN"])
        NP = NP + Counter(sessions[session]["NP"])
        
        total = Counter(sessions[session]["NN"])+Counter(sessions[session]["NN"])
        max_word_count = max(max_word_count, sum(total.values()))

    print("max sentence length", max_sentence_length)
    print("max word count", max_word_count)
    print("Nurse", N, NN)
    print("Patient", P, NP)
    
    nn, np = {}, {}
    for key in NN:
        nn[key] = NN[key]/N[key]
    for key in NP:
        np[key] = NP[key]/P[key]

    print("Nurse Length", nn)
    print("Patient Length", np)

    return sessions

def text_tiling(batch, aid):
    # initialize the tokenizer
    
    start_id = 0
    
    temp = batch
    sent_list = [ ("Nurse: "+x["content"] if x["speaker"]=="N" else "Patient: "+x["content"]).strip() for x in temp ]
    labels = { ("Nurse: "+x["content"] if x["speaker"]=="N" else "Patient: "+x["content"]).strip():dictionary[x["code"]] for x in temp }
    
    tt = tiling.TextTilingTokenizer()
    s = "\n\n".join(sent_list)
    result = tt.tokenize(s)         
    
    #result = sent_list

    entries = []
    for p in result:
        sents = p.split("\n\n")
        #print(sents)
        sents = [ x for x in sents if x.strip()!=""]
       
                
        toAdd = [ [] ] 
                
        while sents:
            curr = sents.pop(0)
            leng = len(" ".join(curr).split())

            if len(" ".join(toAdd[-1]).split()) + leng < MAX:
                toAdd[-1].append(curr)
            else:
                toAdd.append([curr])
        
        entries += toAdd
    
    res = []

    for e in entries:
        if not e:
            continue
        
        labs = [labels[x.strip()] for x in e ] 
        assert(len(labs) == len(e))

        entry = {"abstract_id":aid, "sentences":e,"labels":labs, "confs":[1.0 for x in range(len(labs))] }
        res.append(entry)
        
        aid += 1
            
        #print(entry)

    return res, aid

def writeToJSONL(json_path, jsonl_path):
    jsonl_path = jsonl_path+"/level-"+str(level)+"/"
    #json_path = "data/json_data"
    #jsonl_path = "jsonl"

    if not os.path.exists(jsonl_path):
        os.mkdir(jsonl_path)
    
   
    sessions = []
    for file_path in glob.glob('{}/*.json'.format(json_path)):
        with open(file_path) as json_file:
            temp = json.load(json_file)       
            temp = list(temp.values())
        sessions.append(temp)
    N = len(sessions)
    
    print("Number of Sessions read: ", N)

    #random.shuffle(sessions)
    sessions = np.array(sessions)
    S = []
    kf = model_selection.KFold(n_splits = k, shuffle = True, random_state=None)
    for train_index, test_index in  (kf.split(sessions)):
        #print(train_index, test_index)
        #print(sessions[test_index])
        S.append([sessions[train_index], sessions[test_index]])
    

    for ifold, session in enumerate(S):
        TRAIN = session[0]
        TEST = session[1]
        
        count, aid = 0, 0
        train, dev, test = [], [], []

        DEV = int(len(TRAIN)* 0.9)
        for temp in TRAIN:
            #"""
            entries, aid  = text_tiling(temp, aid)
            for e in entries:   
                if count < DEV:
                    train.append(e)
                else:
                    dev.append(e)
            count += 1
            #"""

            """
            start_id = 0
            while start_id < len(temp):
                toAdd = []
                for i in range(start_id, min(start_id + length, len(temp))):
                    toAdd.append(temp[i])
                
                start_id += 1
                sentences = [ "Nurse: "+x["content"] if x["speaker"]=="N" else "Patient: "+x["content"] for x in toAdd]
                labels = [ dictionary[x["code"]] for x in toAdd ]
                aid += 1    

                confs = [ 1.0 for x in toAdd]
                entry = {"abstract_id":aid, "sentences":sentences,"labels":labels , "confs":confs  }
                
                assert(len(sentences) == len(labels))

                if count < DEV:
                    train.append(entry)
                else:
                    dev.append(entry)
            count += 1
            """
        for temp in TEST:  
            entries, aid =  text_tiling(temp, aid)
            for e in entries:   
                test.append(e)                        
                      
            count += 1

        with open(jsonl_path+"/train-"+str(ifold)+".jsonl", "w") as json_file:
            for item in train:
                json.dump(item, json_file)
                json_file.write("\n")

        with open(jsonl_path+"/dev-"+str(ifold)+".jsonl", "w") as json_file:
            for item in dev:
                json.dump(item, json_file)
                json_file.write("\n")

        with open(jsonl_path+"/test-"+str(ifold)+".jsonl", "w") as json_file:
            for item in test:
                json.dump(item, json_file)
                json_file.write("\n")   
    return


  
def readFromJSON(json_path) -> list:
    data = []

    for file_path in glob.glob('{}/*.json'.format(json_path)):
        with open(file_path) as json_file:
            temp = json.load(json_file)
            for item in temp:
                data.append(temp[item])

    return data


def testGen(json_path, jsonl_path):
    dictionary = five
    jsonl_path = "./example/"
    json_path = "./example/one_out/"

    if not os.path.exists(jsonl_path):
        os.mkdir(jsonl_path)
    
   
    sessions = []
    for file_path in glob.glob('{}/*.json'.format(json_path)):
        with open(file_path) as json_file:
            temp = json.load(json_file)       
            temp = list(temp.values())
        sessions.append(temp)
    N = len(sessions)
    print("Number of Sessions read: ", N)
    sessions = np.array(sessions)
 
   

    for ifold, session in enumerate(sessions):

        TEST = [session]
        
        count, aid = 0, 0
        train, dev, test = [], [], []

        #print(TEST)
        #exit()

        for temp in TEST:  
            start_id = 0
            while start_id < len(temp):
                toAdd = []
                for i in range(start_id, min(start_id + length, len(temp))):
                    toAdd.append(temp[i])
                
                start_id += 1#length
                sentences = [ "Nurse: "+x["content"] if x["speaker"]=="N" else "Patient: "+x["content"] for x in toAdd]
                labels = [ dictionary[x["code"]] for x in toAdd ]
                aid += 1    

                confs = [ 1.0 for x in toAdd]
                entry = {"abstract_id":aid, "sentences":sentences,"labels":labels , "confs":confs  }
                
                assert(len(sentences) == len(labels))

                test.append(entry)
            count += 1       



        with open(jsonl_path+"/test-"+str(ifold)+".jsonl", "w") as json_file:
            for item in test:
                json.dump(item, json_file)
                json_file.write("\n")   
    return
####################################
####################################
####################################

def f_then_ts_writeToJSONL(json_path, jsonl_path):
    jsonl_path = jsonl_path+"/level-"+str(level)+"/"
    #json_path = "data/json_data"
    #jsonl_path = "jsonl"

    if not os.path.exists(jsonl_path):
        os.mkdir(jsonl_path)
    
   
    sessions = []
    for file_path in glob.glob('{}/*.json'.format(json_path)):
        with open(file_path) as json_file:
            temp = json.load(json_file)       
            temp = list(temp.values())
        sessions.append(temp)
    N = len(sessions)
    print("Number of Sessions read: ", N)

    #random.shuffle(sessions)
    sessions = np.array(sessions)
    S = []
    kf = model_selection.KFold(n_splits = k, shuffle = True, random_state=None)
    for train_index, test_index in  (kf.split(sessions)):
        #print(train_index, test_index)
        #print(sessions[test_index])
        S.append([sessions[train_index], sessions[test_index]])
    

    for ifold, session in enumerate(S):
        TRAIN = session[0]
        TEST = session[1]
        
        count, aid = 0, 0
        train, dev, test = [], [], []

        DEV = int(len(TRAIN)* 0.9)
        for temp in TRAIN:  
            start_id = 0
            while start_id < len(temp):
                toAdd = []
                for i in range(start_id, min(start_id + length, len(temp))):
                    toAdd.append(temp[i])
                
                start_id += 1
                sentences = [ "Nurse: "+x["content"] if x["speaker"]=="N" else "Patient: "+x["content"] for x in toAdd]
                labels = [ dictionary[x["code"]] for x in toAdd ]
                aid += 1    

                confs = [ 1.0 for x in toAdd]
                entry = {"abstract_id":aid, "sentences":sentences,"labels":labels , "confs":confs  }
                
                assert(len(sentences) == len(labels))

                if count < DEV:
                    train.append(entry)
                else:
                    dev.append(entry)
            count += 1
        
        # initialize the tokenizer
        #tt = tiling.TextTilingTokenizer()

        for temp in TEST:  
            start_id = 0
            #"""
            entries, aid =  text_tiling(temp, aid)
            for e in entries:   
                test.append(e)                        
                      
            count += 1
            #"""
            
           
        with open(jsonl_path+"/train-"+str(ifold)+".jsonl", "w") as json_file:
            for item in train:
                json.dump(item, json_file)
                json_file.write("\n")

        with open(jsonl_path+"/dev-"+str(ifold)+".jsonl", "w") as json_file:
            for item in dev:
                json.dump(item, json_file)
                json_file.write("\n")

        with open(jsonl_path+"/test-"+str(ifold)+".jsonl", "w") as json_file:
            for item in test:
                json.dump(item, json_file)
                json_file.write("\n")   
    return

####################################
####################################
####################################
parser = English()
def tokenize2(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens



def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

en_stop = set(nltk.corpus.stopwords.words('english'))

####################################
####################################
####################################


####################################
####################################
####################################

def fixed_writeToJSONL(json_path, jsonl_path):
    jsonl_path = jsonl_path+"/level-"+str(level)+"/"
    #json_path = "data/json_data"
    #jsonl_path = "jsonl"

    #ldamodel = gensim.models.LdaModel.load('./gen_data/multicore_model5_8.gensim')
    #lda_dictionary = corpora.Dictionary.load('./gen_data/multicore_model5_8.gensim.id2word')


    if not os.path.exists(jsonl_path):
        os.mkdir(jsonl_path)
    
   
    sessions = []
    for file_path in glob.glob('{}/*.json'.format(json_path)):
        with open(file_path) as json_file:
            temp = json.load(json_file)       
            temp = list(temp.values())
        sessions.append(temp)
    N = len(sessions)
    print("Number of Sessions read: ", N)

    #random.shuffle(sessions)
    sessions = np.array(sessions)
    S = []
    kf = model_selection.KFold(n_splits = k, shuffle = True, random_state=None)
    for train_index, test_index in  (kf.split(sessions)):
        #print(train_index, test_index)
        #print(sessions[test_index])
        S.append([sessions[train_index], sessions[test_index]])
    

    for ifold, session in enumerate(S):
        TRAIN = session[0]
        TEST = session[1]
        
        count, aid = 0, 0
        train, dev, test = [], [], []

        DEV = int(len(TRAIN)* 0.9)
        for temp in TRAIN:  
            start_id = 0
            while start_id < len(temp):
                toAdd = []
                for i in range(start_id, min(start_id + length, len(temp))):
                    toAdd.append(temp[i])
                
                start_id += 1
                sentences = [ "Nurse: "+x["content"] if x["speaker"]=="N" else "Patient: "+x["content"] for x in toAdd]
                labels = [ dictionary[x["code"]] for x in toAdd ]
                
                ####
                """
                topics = [ prepare_text_for_lda(x["content"]) for x in toAdd ]
                topics = [ lda_dictionary.doc2bow(x) for x in topics ]
                topics = [ ldamodel.get_document_topics(x) for x in topics ] 
                topics = [ sorted(x, key = lambda x: x[1])[-1][0] for x in topics ] 

                assert(len(topics) == len(sentences))
                for i in range(len(topics)):
                    sentences[i] = "Topic: "+str(topics[i])+ ", "+ sentences[i]
                """
                ####

                aid += 1    

                confs = [ 1.0 for x in toAdd]
                entry = {"abstract_id":aid, "sentences":sentences,"labels":labels , "confs":confs  }
                
                assert(len(sentences) == len(labels))

                if count < DEV:
                    train.append(entry)
                else:
                    dev.append(entry)
            count += 1
        
        # initialize the tokenizer
        #tt = tiling.TextTilingTokenizer()

        for temp in TEST:  
            start_id = 0
            
            while start_id < len(temp):
                toAdd = []
                for i in range(start_id, min(start_id + length, len(temp))):
                    toAdd.append(temp[i])
                
                start_id += length
                sentences = [ "Nurse: "+x["content"] if x["speaker"]=="N" else "Patient: "+x["content"] for x in toAdd]
                labels = [ dictionary[x["code"]] for x in toAdd ]
                aid += 1    
 
                ####
                """
                topics = [ prepare_text_for_lda(x["content"]) for x in toAdd ]
                topics = [ lda_dictionary.doc2bow(x) for x in topics ]
                topics = [ ldamodel.get_document_topics(x) for x in topics ] 
                topics = [ sorted(x, key = lambda x: x[1])[-1][0] for x in topics ] 

                assert(len(topics) == len(sentences))
                for i in range(len(topics)):
                    sentences[i] = "Topic: "+str(topics[i])+ ", "+ sentences[i]
                """
                ####
                
                confs = [ 1.0 for x in toAdd]
                entry = {"abstract_id":aid, "sentences":sentences,"labels":labels , "confs":confs  }
                
                

                assert(len(sentences) == len(labels))

                test.append(entry)
            count += 1       
            

        with open(jsonl_path+"/train-"+str(ifold)+".jsonl", "w") as json_file:
            for item in train:
                json.dump(item, json_file)
                json_file.write("\n")

        with open(jsonl_path+"/dev-"+str(ifold)+".jsonl", "w") as json_file:
            for item in dev:
                json.dump(item, json_file)
                json_file.write("\n")

        with open(jsonl_path+"/test-"+str(ifold)+".jsonl", "w") as json_file:
            for item in test:
                json.dump(item, json_file)
                json_file.write("\n")   
    return


def main():
    txt_path = "./aws/"
    jsonl_path = "./aws/"


    fixed_writeToJSONL(txt_path, jsonl_path)
      
    return

main()
