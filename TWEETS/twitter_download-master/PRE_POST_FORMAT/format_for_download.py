import argparse

parser = argparse.ArgumentParser(description="specify file to format")
parser.add_argument("input", help="specify input tsv file")
parser.add_argument("output", help="specify output tsv file")
args = parser.parse_args()

with open(args.input, 'r') as file, open(args.output, 'w') as output:
    for line in file:
        res = [ele for ele in line.split('\t')]
        formatted_line = res[0] + '\t' + res[1] + '\n'
        output.write(formatted_line)
    