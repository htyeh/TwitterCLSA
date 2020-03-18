#!/usr/local/bin/python3
import os
import re

emoticon_dict = {
        ":-)":"smiley",
        ":-]":"smiley",
        ":-3":"smiley",
        ":->":"smiley",
        "8-)":"smiley",
        ":-}":"smiley",
        ":)":"smiley",
        ":]":"smiley",
        ":3":"smiley",
        ":>":"smiley",
        "8)":"smiley",
        ":}":"smiley",
        ":o)":"smiley",
        ":c)":"smiley",
        ":^)":"smiley",
        "=]":"smiley",
        "=)":"smiley",
        ":-))":"smiley",
        ":-D":"smiley",
        "8-D":"smiley",
        "x-D":"smiley",
        "X-D":"smiley",
        ":D":"smiley",
        "8D":"smiley",
        "xD":"smiley",
        "XD":"smiley",
        ":-(": "sad",
        ":-c":"sad",
        ":-<":"sad",
        ":-[":"sad",
        ":(":"sad",
        ":c":"sad",
        ":<":"sad",
        ":[":"sad",
        ":-||":"sad",
        ">:[":"sad",
        ":{":"sad",
        ":@":"sad",
        ">:(":"sad",
        ":'-(":"sad",
        ":'(":"sad",
        ":-P":"playful",
        "X-P":"playful",
        "x-p":"playful",
        ":-p":"playful",
        ":-Þ":"playful",
        ":-þ":"playful",
        ":-b":"playful",
        ":P":"playful",
        "XP":"playful",
        "xp":"playful",
        ":p":"playful",
        ":Þ":"playful",
        ":þ":"playful",
        ":b":"playful",
        "<3":"love"
        }

def clean_tweet(tweet):
        tweet = tweet.lower()
        tweet = ' '.join(re.sub("(\w+:\/\/\S+)", " ", tweet).split())
        tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)", " ", tweet).split())
        tweet = ' '.join([emoticon_dict[word] if word in emoticon_dict else word for word in tweet.split()])
        tweet = ' '.join(re.sub("[\.\,\!\?\:\;\-\=\"]", " ", tweet).split())
        return tweet

# cp full files from FULL/ first
files_to_clean = [file for file in os.listdir('.') if file.endswith('.tsv')]
print('clearning:', ', '.join(files_to_clean))
for original_file in files_to_clean:
        clean_lines = []
        with open(original_file) as in_file:
                for line in in_file:
                        res = [ele.strip() for ele in line.split('\t')]
                        clean_line = res[0] + '\t' + res[1] + '\t' + clean_tweet(res[2]) + '\n'
                        clean_lines.append(clean_line)
        with open('CLEAN/' + original_file, 'w') as out_file:
                for line in clean_lines:
                        out_file.write(line)


# max_len = 0
# en_files = [file for file in os.listdir('.') if file.startswith('en')]
# for file in en_files:
#     with open(file) as en_file:
#         for line in en_file:
#             res = [ele.strip() for ele in line.split('\t')]
#             if len(res[2].split()) > max_len:
#                 max_len = len(res[2].split())
# print(max_len)