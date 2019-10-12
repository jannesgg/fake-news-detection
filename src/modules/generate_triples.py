from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-f", "--file", dest="filename", help="write report to FILE", metavar="FILE"
    )
    args = parser.parse_args()

    st = StanfordNERTagger(
        "./stanford-ner-2018-10-16/classifiers/english.conll.4class.distsim.crf.ser.gz",
        "./stanford-ner-2018-10-16/stanford-ner.jar",
        encoding="utf-8",
    )

    text = pd.read_csv(args.filename, index_col=None, header=0)["text"]

    tokenized_text = [word_tokenize(i) for i in text]
    classified_text = [st.tag(i) for i in tokenized_text]
