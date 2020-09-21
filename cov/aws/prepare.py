import json
import glob
from collections import Counter
import argparse

import csv
"""
#import tscribe
corpus_root = "./orig"
path = sorted(glob.glob('{}/*.txt'.format(corpus_root)))
for file_path in path:
    with open(file_path, "r", encoding="ISO-8859-1") as fh:
        with open("./"+file_path[6:-4]+"_ref.txt", "w") as nfh:
            for line in fh:
                utterance = ' '.join(line.strip().split()[1:])
                
                #exit()
                #print(utterance)
                if utterance:
                    speaker = line.strip().split()[0]
                    nfh.write(speaker+": "+utterance)
                    nfh.write("\n")
                    #nfh.write("\n")
"""
corpus_root = "./"
path = sorted(glob.glob('{}/*.csv'.format(corpus_root)))
for file_path in path:
    with open(file_path, "r", encoding="ISO-8859-1") as fh:
        with open("./"+file_path[:-4]+".txt", "w") as nfh:
            csv_reader = csv.reader(fh, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                #print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    nfh.write(row[3]+": "+row[4]+"\n")
                line_count += 1
exit()

corpus_root = "./google"
path = sorted(glob.glob('{}/*.json'.format(corpus_root)))
for file_path in path:
    with open(file_path, "r", encoding="ISO-8859-1") as fh:
        with open("./google_"+file_path[9:-4]+"_google.txt", "w") as nfh:
            tr = json.load(fh)
            for seg in tr["results"]: #[0]["alternatives"]:
                if "transcript" not in seg["alternatives"][0]:
                    break
                utterance = seg["alternatives"][0]["transcript"].strip()
                #utterance = (seg["alternatives"]["transcript"])
                #print(utterance)
                nfh.write(utterance)
                nfh.write("\n")
            #exit()
            """
            for line in fh:
                utterance = ' '.join(line.strip().split()[1:])
                #exit()
                #print(utterance)
                if utterance:
                    nfh.write(utterance)
                    nfh.write(" ")
                    #nfh.write("\n")
            """

    #save_path=file_path[:-5]+".csv"
    #print("Diarizing",file_path)
    #print("Saving to", save_path)

    #tscribe.write(file_path, save_as=save_path, format="csv")

#exit()

from jiwer import wer

orig_path = sorted(glob.glob('{}/grade*_cleaned.txt'.format("./")))
aws_path = sorted(glob.glob('{}/aws*_cleaned.txt'.format("./")))
google_path = sorted(glob.glob('{}/google*_cleaned.txt'.format("./")))

print(orig_path)
print(aws_path)
print(google_path)

for o, a, g in zip(orig_path, aws_path, google_path):
    print(o,a,g)
    with open(o, "r") as of:
        os = [ x.strip() for x in (of.readlines())]
        #print(os)
        #exit()
    with open(a, "r") as af:
        aws = [ x.strip() for x in (af.readlines())]
    with open(g, "r") as gf:
        gs = [ x.strip() for x in (gf.readlines())]

    aws_error = wer(os, aws)
    gs_error = wer(os, gs)
    print(aws_error, gs_error)
