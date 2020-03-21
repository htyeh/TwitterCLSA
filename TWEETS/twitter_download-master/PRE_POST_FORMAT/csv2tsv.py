import csv

with open('Mozetic_DE.csv','r') as csvin, open('Mozetic_DE.tsv', 'w') as tsvout:
    csvin = csv.reader(csvin)
    tsvout = csv.writer(tsvout, delimiter='\t')

    for row in csvin:
        tsvout.writerow(row)