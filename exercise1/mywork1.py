import nltk
import json
import re

def file_read():
    with open('mytest.jsonl','r') as f:
    # with open('signal-news1.jsonl','r') as f:
        while True:
            line = f.readline()
            if line:
                data = json.loads(line).get('content')
                return data



