import os
import re
import numpy as np
import pandas as pd

import nltk
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from argparse import ArgumentParser
import json
from collections import Counter
from nltk.corpus import stopwords
import spacy

nlp = spacy.load("en_core_web_sm")

# paths

model_path = "../../models/Stanford-OpenIE-Python/"

# functions


def get_root_verb(text):
    doc = nlp(text)
    if len(doc) > 0:
        for token in doc:
            if token.dep_ == "ROOT" and token.head.pos_ == "VERB":
                return str(token)
            else:
                return ""
    else:
        return ""


def extract_compounds(text):
    """Extract compound noun phrases with beginning and end idxs.

    Keyword arguments:
    text -- the actual text source from which to extract entities

    """
    comp_idx = 0
    compound = []
    compound_nps = []
    tok_idx = 0
    for idx, tok in enumerate(nlp(text)):
        if tok.dep_ == "compound":

            # capture hyphenated compounds
            children = "".join([c.text for c in tok.children])
            if "-" in children:
                compound.append("".join([children, tok.text]))
            else:
                compound.append(tok.text)

            # remember starting index of first child in compound or word
            try:
                tok_idx = [c for c in tok.children][0].idx
            except IndexError:
                if len(compound) == 1:
                    tok_idx = tok.idx
            comp_idx = tok.i

        # append the last word in a compound phrase
        if tok.i - comp_idx == 1:
            compound.append(tok.text)
            if len(compound) > 1:
                compound = " ".join(compound)
                compound_nps.append(
                    (compound, tok_idx, tok_idx + len(compound), "COMPOUND")
                )

            # reset parameters
            tok_idx = 0
            compound = []

    if len(compound_nps) != 0:
        return compound_nps[0][0]
    else:
        return ""


def fix_entities(text):
    if "clinton" in text:
        return "hillary clinton"
    elif "donald" in text:
        return "donald trump"
    elif "hillary" in text:
        return "hillary clinton"
    elif "trump" in text:
        return "donald trump"
    else:
        return text


def shorten_relations(relation, n=3):
    if len(word_tokenize(relation)) > n:
        return ""
    else:
        return relation


def remove_stops(word):
    if word.lower() in set(stopwords.words("english")):
        return ""
    else:
        return word


def extract_entities_spacy(text):
    proc = nlp(text)
    if len(proc.ents) == 0:
        return ""
    else:
        return " ".join([x.text for x in proc.ents])


def pp_pipeline(triples_path, output_dir, model_name):
    """Perform pre-processing pipeline"""

    filenames = [
        triples_path + i
        for i in sorted(os.listdir(triples_path))
        if ".txt" in i and "_triple_" in i
    ]

    swedish_ids = []
    doc_ids = []

    with open(output_dir + "{:s}.txt".format(model_name), "w") as outfile:
        for j in range(len(filenames)):
            with open(filenames[j]) as infile:
                m = re.search("\d", filenames[j])
                if filenames[j][m.start()] == "0" and "swedish" not in filenames[j]:
                    for line in infile:
                        outfile.write(
                            "en" + "|" + "true" + "|" + str(j) + "|" + line + "\n"
                        )
                elif filenames[j][m.start()] != "0" and "swedish" not in filenames[j]:
                    for line in infile:
                        outfile.write(
                            "en" + "|" + "fake" + "|" + str(j) + "|" + line + "\n"
                        )
                elif "swedish" in filenames[j]:
                    doc_ids.append(("".join(filter(str.isdigit, filenames[j])))[1:])
                    swedish_ids.append(j)
                    for line in infile:
                        outfile.write(
                            "sv" + "|" + "fake" + "|" + str(j) + "|" + line + "\n"
                        )

    np.savetxt(
        "../../data/external/swedish_news/ids.txt", np.array(swedish_ids), fmt="%s"
    )
    np.savetxt(
        "../../data/external/swedish_news/doc_ids.txt", np.array(doc_ids), fmt="%s"
    )

    df_triples = pd.read_csv(
        output_dir + "{:s}.txt".format(model_name), header=None, sep="\t"
    )

    df_triples = pd.DataFrame(
        df_triples[0].str.split("|", 5).tolist(),
        columns=["lang", "status", "article_id", "e1", "r", "e2"],
    )

    lemmatizer = WordNetLemmatizer()

    df_triples["l1"] = (
        df_triples["e1"]
        .apply(lambda x: extract_entities_spacy(x))
        .apply(lambda x: x.lower().strip())
        .apply(lambda x: fix_entities(x))
        .apply(remove_stops)
        .apply(lambda x: re.sub("[^\s'_A-Za-z]", "", x))
        .apply(lambda x: x.lstrip().rstrip())
    )

    df_triples["l2"] = (
        df_triples["e2"]
        .apply(lambda x: extract_entities_spacy(x))
        .apply(lambda x: x.lower().strip())
        .apply(lambda x: fix_entities(x))
        .apply(remove_stops)
        .apply(lambda x: re.sub("[^\s'_A-Za-z]", "", x))
        .apply(lambda x: x.lstrip().rstrip())
    )

    df_triples["rel"] = (
        df_triples["r"]
        .apply(lambda x: x.lower().strip())
        .apply(lambda x: lemmatizer.lemmatize(x, pos="v"))
        .apply(lambda x: re.sub("[^\s'_A-Za-z]", "", x))
        .apply(lambda x: x.lstrip().rstrip())
    )

    total_entities = pd.concat([df_triples["l1"], df_triples["l2"]])
    c = Counter(total_entities)
    unique_entities = pd.Series(list(c.keys()))

    total_relations = pd.Series(df_triples["rel"])
    rc = Counter(total_relations)
    unique_relations = pd.Series(list(rc.keys()))

    d = {
        k: v for k, v in zip(unique_entities, [i for i in range(len(unique_entities))])
    }

    d_sorted = sorted(d.items(), key=lambda kv: kv[1])

    r = {
        k: v
        for k, v in zip(unique_relations, [i for i in range(len(unique_relations))])
    }

    r_sorted = sorted(r.items(), key=lambda kv: kv[1])

    with open(output_dir + "relation2id.txt", "w") as f:
        f.write("{:d}\n".format(len(r.items())))
        for i in range(len(r_sorted)):
            f.write("{:s}\t{:d}\n".format(r_sorted[i][0], i))

    with open(output_dir + "entity2id.txt", "w") as f:
        f.write("{:d}\n".format(len(d.items())))
        for i in range(len(d_sorted)):
            f.write("/m/{:s}\t{:d}\n".format(d_sorted[i][0], i))

    df_triples = df_triples.sort_values(by="status").replace("", np.nan).dropna()

    df_fake = df_triples[
        (df_triples["status"] == "fake") & (df_triples["lang"] == "en")
    ].reset_index()
    df_true = df_triples[
        (df_triples["status"] == "true") & (df_triples["lang"] == "en")
    ].reset_index()

    fake_ids = sorted([int(i) for i in list(set(df_fake["article_id"]))])
    true_ids = sorted([int(i) for i in list(set(df_true["article_id"]))])

    np.random.seed(42)
    np.random.shuffle(fake_ids)
    np.random.shuffle(true_ids)

    train = df_fake[df_fake["article_id"].astype(int).isin(fake_ids[:600])]

    validation = train

    test = df_triples[(df_triples["lang"] == "sv")]

    with open(output_dir + "entities.txt", "w") as file:
        file.write(json.dumps(d_sorted))

    with open(output_dir + "relations.txt", "w") as file:
        file.write(json.dumps(r_sorted))

    train.to_csv(output_path + "train_set.csv")
    validation.to_csv(output_path + "validation_set.csv")
    test.to_csv(output_path + "test_set.csv")

    validation = validation.reset_index()
    test = test.reset_index()
    t = test
    v = validation

    with open(output_dir + "train2id.txt", "w") as f:
        f.write("{:d}\n".format(len(train)))
        for a, b, c in zip(train["l1"], train["l2"], train["rel"]):
            try:
                f.write("{:d} {:d} {:d}\n".format(d[a], d[b], r[c]))
            except KeyError:
                pass

    rem_ind_v = []

    with open(output_dir + "valid2id.txt", "w") as f:
        f.write("{:d}\n".format(len(validation)))
        for i, a, b, c in zip(
            validation.index, validation["l1"], validation["l2"], validation["rel"]
        ):
            try:
                f.write("{:d} {:d} {:d}\n".format(d[a], d[b], r[c]))
            except KeyError:
                rem_ind_v.append(i)

    v.drop(v.index[list(set(rem_ind_v))], inplace=True)

    rem_ind = []

    with open(output_dir + "test2id.txt", "w") as f:
        f.write("{:d}\n".format(len(test)))
        for i, a, b, c in zip(test.index, test["l1"], test["l2"], test["rel"]):
            try:
                f.write("{:d} {:d} {:d}\n".format(d[a], d[b], r[c]))
            except KeyError:
                rem_ind.append(i)

    t.drop(t.index[list(set(rem_ind))], inplace=True)

    print(len(set(t["article_id"])))

    ### fix first lines of text files
    with open(output_dir + "train2id.txt") as f:
        lines = f.readlines()
    lines[0] = str(len(lines) - 1) + "\n"
    with open(output_dir + "train2id.txt", "w") as f:
        f.writelines(lines)

    with open(output_dir + "valid2id.txt") as f:
        lines = f.readlines()
    lines[0] = str(len(lines) - 1) + "\n"
    with open(output_dir + "valid2id.txt", "w") as f:
        f.writelines(lines)

    with open(output_dir + "test2id.txt") as f:
        lines = f.readlines()
    lines[0] = str(len(lines) - 1) + "\n"
    with open(output_dir + "test2id.txt", "w") as f:
        f.writelines(lines)

    v.to_csv(output_dir + "validation_set.csv")

    t.to_csv(output_dir + "test_set.csv")


def n_n(output_path):
    """Setup the training, validation and test sets in the required format"""
    lef = {}
    rig = {}
    rellef = {}
    relrig = {}

    triple = open(output_path + "train2id.txt", "r")
    valid = open(output_path + "valid2id.txt", "r")
    test = open(output_path + "test2id.txt", "r")

    ls = triple.readlines()
    tot = len(ls) - 1

    # (int)(triple.readline())
    for i in range(tot):
        content = ls[i + 1]
        # content = triple.readline()
        h, t, r = content.strip().split()
        if not (h, r) in lef:
            lef[(h, r)] = []
        if not (r, t) in rig:
            rig[(r, t)] = []
        lef[(h, r)].append(t)
        rig[(r, t)].append(h)
        if not r in rellef:
            rellef[r] = {}
        if not r in relrig:
            relrig[r] = {}
        rellef[r][h] = 1
        relrig[r][t] = 1

    ls = valid.readlines()
    tot = len(ls) - 1
    # (int)(valid.readline())
    for i in range(tot):
        content = ls[i + 1]
        # content = valid.readline()
        h, t, r = content.strip().split()
        if not (h, r) in lef:
            lef[(h, r)] = []
        if not (r, t) in rig:
            rig[(r, t)] = []
        lef[(h, r)].append(t)
        rig[(r, t)].append(h)
        if not r in rellef:
            rellef[r] = {}
        if not r in relrig:
            relrig[r] = {}
        rellef[r][h] = 1
        relrig[r][t] = 1

    ls = test.readlines()
    tot = len(ls) - 1
    # (int)(test.readline())
    for i in range(tot):
        content = ls[i + 1]
        # content = test.readline()
        h, t, r = content.strip().split()
        if not (h, r) in lef:
            lef[(h, r)] = []
        if not (r, t) in rig:
            rig[(r, t)] = []
        lef[(h, r)].append(t)
        rig[(r, t)].append(h)
        if not r in rellef:
            rellef[r] = {}
        if not r in relrig:
            relrig[r] = {}
        rellef[r][h] = 1
        relrig[r][t] = 1

    test.close()
    valid.close()
    triple.close()

    f = open(output_path + "type_constrain.txt", "w")
    f.write("%d\n" % (len(rellef)))
    for i in rellef:
        f.write("%s\t%d" % (i, len(rellef[i])))
        for j in rellef[i]:
            f.write("\t%s" % (j))
        f.write("\n")
        f.write("%s\t%d" % (i, len(relrig[i])))
        for j in relrig[i]:
            f.write("\t%s" % (j))
        f.write("\n")
    f.close()

    rellef = {}
    totlef = {}
    relrig = {}
    totrig = {}

    for i in lef:
        if not i[1] in rellef:
            rellef[i[1]] = 0
            totlef[i[1]] = 0
        rellef[i[1]] += len(lef[i])
        totlef[i[1]] += 1.0

    for i in rig:
        if not i[0] in relrig:
            relrig[i[0]] = 0
            totrig[i[0]] = 0
        relrig[i[0]] += len(rig[i])
        totrig[i[0]] += 1.0

    s11 = 0
    s1n = 0
    sn1 = 0
    snn = 0
    f = open(output_path + "test2id.txt", "r")
    ls = f.readlines()
    tot = len(ls) - 1
    # tot = (int)(f.readline())
    for i in range(tot):
        content = ls[i + 1]
        # content = f.readline()
        h, t, r = content.strip().split()
        rign = rellef[r] / totlef[r]
        lefn = relrig[r] / totrig[r]
        if rign <= 1.5 and lefn <= 1.5:
            s11 += 1
        if rign > 1.5 and lefn <= 1.5:
            s1n += 1
        if rign <= 1.5 and lefn > 1.5:
            sn1 += 1
        if rign > 1.5 and lefn > 1.5:
            snn += 1
    f.close()

    f = open(output_path + "test2id.txt", "r")
    f11 = open(output_path + "1-1.txt", "w")
    f1n = open(output_path + "1-n.txt", "w")
    fn1 = open(output_path + "n-1.txt", "w")
    fnn = open(output_path + "n-n.txt", "w")
    fall = open(output_path + "test2id_all.txt", "w")

    ls = f.readlines()
    tot = len(ls) - 1

    # tot = (int)(f.readline())
    fall.write("%d\n" % (tot))
    f11.write("%d\n" % (s11))
    f1n.write("%d\n" % (s1n))
    fn1.write("%d\n" % (sn1))
    fnn.write("%d\n" % (snn))
    for i in range(tot):
        content = ls[i + 1]
        # content = f.readline()
        h, t, r = content.strip().split()
        rign = rellef[r] / totlef[r]
        lefn = relrig[r] / totrig[r]
        if rign <= 1.5 and lefn <= 1.5:
            f11.write(content)
            fall.write("0" + "\t" + content)
        if rign > 1.5 and lefn <= 1.5:
            f1n.write(content)
            fall.write("1" + "\t" + content)
        if rign <= 1.5 and lefn > 1.5:
            fn1.write(content)
            fall.write("2" + "\t" + content)
        if rign > 1.5 and lefn > 1.5:
            fnn.write(content)
            fall.write("3" + "\t" + content)
    fall.close()
    f.close()
    f11.close()
    f1n.close()
    fn1.close()
    fnn.close()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-m", "--model_name", dest="mod", help="write report to FILE", metavar="FILE"
    )

    args = parser.parse_args()

    # run pipelines

    output_path = "../../data/processed/cv-models/{:s}/".format(args.mod)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    triples_path = output_path + "small_files/"

    if not os.path.exists(triples_path):
        os.mkdir(triples_path)

    print("Processing triples...")

    pp_pipeline(triples_path, output_path, args.mod)

    print("Processing...done")

    n_n(output_path)
