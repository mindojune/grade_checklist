import json
import glob
from collections import Counter
import argparse
import random
import re
import os
import jsonlines

def tests_to_jsonl():

    #names =  [x.strip(",") for x in "mft_food, mft_sport, mft_nondrug, mft_drug, inv_n, inv_d, dir_nd, dir_dn".split() ]
    #names =  [x.strip(",") for x in "mft_food, mft_sport, mft_nondrug, mft_drug, inv_food_punct, inv_food_typo".split() ]

    #print(names)
    
    names = ["aws", "google", "test"]
    for name in names:
        with open("./tests/"+name+".txt", "r") as fh:
            data = fh.read().splitlines()

        fname = "./tests/"+name+".jsonl"
        if os.path.isfile(fname):
            os.remove(fname)
        with jsonlines.open(fname, "a") as writer:
            for datum in data:
                entry = {"abstrac_id": 0}
                entry["sentences"] = [datum]
                entry["labels"] = [0]
                entry["confs"] = [0]

                writer.write(entry)
    
 tests_to_jsonl()
