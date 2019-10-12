from multiprocessing.dummy import Pool
import os
import subprocess
import re
import numpy as np
import pandas as pd

from nltk import WordNetLemmatizer
from argparse import ArgumentParser
from functools import partial
from multiprocessing.dummy import Pool
import time
import json


input_path = "../data/external/isot_dataset/"
model_path = "../models/Stanford-OpenIE-Python/"
triples_path = "../data/processed/test_set/small_files/"

commands = [
    model_path
    + "process_large_corpus.sh "
    + triples_path
    + i
    + " "
    + triples_path
    + "_triple_"
    + i
    for i in os.listdir(triples_path)
    if ".txt" in i
]

if __name__ == "__main__":

    pool = Pool(4)  # four concurrent commands at a time

    for i, returncode in enumerate(
        pool.imap(partial(subprocess.call, shell=True), commands)
    ):
        if returncode != 0:
            print("%d command failed: %d" % (i, returncode))
        else:
            if i % 100 == 0:
                print(
                    "Triples progress: "
                    + str(np.round(i / len(os.listdir(triples_path)), 2))
                )
