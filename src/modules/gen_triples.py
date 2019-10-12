import os
import pandas as pd

from argparse import ArgumentParser
from pathlib import Path
from pycorenlp import *
from tqdm import tqdm

# paths
base = Path("../..")
input_path = Path(base, "data/external/isot_dataset/")
model_path = Path(base, "models/Stanford-OpenIE-Python/")
output_base = Path(base, "data/processed/cv-models/")

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-f", "--file", dest="filename", help="write report to FILE", metavar="FILE"
    )
    parser.add_argument(
        "-m", "--model_name", dest="mod", help="folder in which to save files", metavar="FILE"
    )

    args = parser.parse_args()

    # Run pipelines
    basename = os.path.basename(Path(input_path, args.filename)).replace(".csv", "")
    output_path = Path(output_base, "{:s}/".format(args.mod))

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    article_df = pd.read_csv(Path(input_path, args.filename)).reset_index()

    # Feed smaller files as input to generate triple files using Stanford OpenIE
    def get_triple(text):

        nlp = StanfordCoreNLP("http://localhost:9000/")
        output = nlp.annotate(text, properties={"annotators": "tokenize,ssplit,pos,depparse,natlog,openie",
                                                "outputFormat": "json",
                                                "openie.triple.strict": "true",
                                                "openie.max_entailments_per_clause": "1"})

        result = output["sentences"][0]["openie"]
        relation_sent = ['', '', '']
        if len(result) > 0:
            for rel in result:
                relation_sent = [rel['subject'], rel['relation'], rel['object']]

        return tuple(relation_sent)


    tqdm.pandas()

    # Limit run to first 1000 articles
    article_df['triple'] = article_df['coref_triples'].iloc[:1000].progress_apply(get_triple)
    article_df.iloc[:1000].to_csv(Path(output_path, args.mod+'.csv'))
