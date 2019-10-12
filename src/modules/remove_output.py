#! /usr/bin/env python3

from json import load, dump

import sys

if len(sys.argv) != 3:
    print("usage: %s INFILE OUTFILE", file=sys.stderr)

with open(sys.argv[1], "rt") as inf:
    ipynb = load(inf)

for ws in ipynb["worksheets"]:
    for cell in ws["cells"]:
        cell["outputs"] = []

with open(sys.argv[2], "wt") as outf:
    dump(ipynb, outf)
