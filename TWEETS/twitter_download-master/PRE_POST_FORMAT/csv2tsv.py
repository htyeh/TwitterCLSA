# python3 csv2tsv.py ids.csv ids.tsv

import csv
import argparse

parser = argparse.ArgumentParser(description="convert csv into tsv format")
parser.add_argument("csv_input", help="specify input csv file")
parser.add_argument("tsv_output", help="specify output tsv file")
args = parser.parse_args()

with open(args.csv_input,'r') as csvin, open(args.tsv_output, 'w') as tsvout:
    csvin = csv.reader(csvin)
    tsvout = csv.writer(tsvout, delimiter='\t')

    for row in csvin:
        tsvout.writerow(row)