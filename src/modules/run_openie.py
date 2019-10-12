MOD_PATH = (
    "/Users/jannes/GU_Modules/Thesis/fake-news-code/models/Stanford-OpenIE-Python/"
)
DATA_PATH = "/Users/jannes/GU_Modules/Thesis/fake-news-code/data/external/all-the-news/"

import subprocess
from argparse import ArgumentParser

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--in_file",
        dest="in_filename",
        help="write report to FILE",
        metavar="FILE",
    )
    parser.add_argument(
        "-fo",
        "--out_file",
        dest="out_filename",
        help="write report to FILE",
        metavar="FILE",
    )
    args = parser.parse_args()

    val = subprocess.call(
        [
            MOD_PATH + "process_large_corpus.sh",
            DATA_PATH + args.in_filename,
            DATA_PATH + args.out_filename,
        ],
        shell=False,
    )
