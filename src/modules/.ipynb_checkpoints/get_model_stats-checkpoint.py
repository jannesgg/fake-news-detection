# imports

import pandas as pd
import networkx as nx
import numpy as np
from nltk import WordNetLemmatizer
import re
from collections import Counter
import os
import matplotlib.pyplot as plt
from operator import itemgetter
from argparse import ArgumentParser

# functions


def prepare_df(test_df, pos_df, neg_df, maxormean="mean"):

    test_df["status"] = np.where(test_df["status"] == "true", 0, 1)
    vals = test_df.groupby("article_id").mean()["status"].reset_index()

    A = pd.concat([test_df.reset_index().drop("index", 1), pos_df, neg_df], axis=1)
    list(test_df.columns).extend(["pos", "neg"])
    cols = list(test_df.columns)
    cols.extend(["pos", "neg"])
    A.columns = cols

    if maxormean == "mean":
        return pd.concat([A.groupby("article_id").mean(), vals.astype(int)], sort=False)
    else:
        return pd.concat([A.groupby("article_id").max(), vals.astype(int)], sort=False)


def compute_stats(grouped, thresh=0.015):

    grouped["tp"] = np.where((grouped["pos"] < thresh) & (grouped["status"] == 1), 1, 0)
    grouped["fp"] = np.where((grouped["pos"] < thresh) & (grouped["status"] == 0), 1, 0)
    grouped["tn"] = np.where((grouped["pos"] > thresh) & (grouped["status"] == 0), 1, 0)
    grouped["fn"] = np.where((grouped["pos"] > thresh) & (grouped["status"] == 1), 1, 0)

    g = grouped[["pos", "neg", "tp", "fp", "tn", "fn", "status"]]

    if g.sum()["tp"] == 0:
        return 0, 0, 0, 0
    else:
        a = (g.sum()["tp"] + g.sum()["tn"]) / (
            g.sum()["tp"] + g.sum()["tn"] + g.sum()["fp"] + g.sum()["fn"]
        )
        p = g.sum()["tp"] / (g.sum()["tp"] + g.sum()["fp"])
        r = g.sum()["tp"] / (g.sum()["tp"] + g.sum()["fn"])
        f = (2 * (p * r)) / (p + r)

        return a, p, r, f


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-m", "--model", dest="model", help="write report to FILE", metavar="FILE"
    )
    args = parser.parse_args()

    validation = pd.read_csv("../../data/processed/validation_set.csv")
    test = pd.read_csv("../../data/processed/test_set.csv")

    ## Validation setup

    valid_embedding_pos = pd.read_csv(
        "../../data/processed/valid_pos_residuals.txt", header=None, sep="\t"
    )
    valid_embedding_neg = pd.read_csv(
        "../../data/processed/valid_neg_residuals.txt", header=None, sep="\t"
    )

    thresholds = np.arange(0.01, 0.5, 0.0005)

    valid_grouped = prepare_df(
        validation, valid_embedding_pos, valid_embedding_neg, "mean"
    )

    a = []
    for i in thresholds:
        output = compute_stats(valid_grouped, i)
        a.append([i, output[0]])

    t = max(a, key=itemgetter(1))[0]

    print("Chosen threshold: " + str(t))

    ## Test setup

    test_embedding_pos = pd.read_csv(
        "../../data/processed/pos_residuals.txt", header=None, sep="\t"
    )
    test_embedding_neg = pd.read_csv(
        "../../data/processed/neg_residuals.txt", header=None, sep="\t"
    )

    test_grouped = prepare_df(test, test_embedding_pos, test_embedding_neg, "mean")

    print("------")
    print("Important Statistics")
    print(compute_stats(test_grouped, t))
