from flask import Flask, request
import urllib.parse
import sys
import re
from copy import deepcopy
from os import mkdir, environ
from os.path import join, exists
from pathlib import Path
import difflib as df
import json
import tensorflow as tf
from ecpp_individual_grammar import read_grammar, fixed_lexed_prog, get_token_list, get_actual_token_list, repair_prog
from predict_eccp_classifier_partials import predict_error_rules
from seq2parse import repair

app = Flask(__name__)


@app.route('/api/text', methods=['GET'])
def get_text():
    url = request.args.get('seq2parse')
    decoded_url = urllib.parse.unquote(url)  # Decode the URL

    # For single (erroneous) file:
    # >>> python seq2parse.py python-grammar.txt ./models 0 input_prog.py
    grammarFile = "python-grammar.txt"
    modelsDir = "./models"
    gpuToUse = '/device:GPU:0'

    max_cost = 5
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    input_prog = decoded_url
    print(input_prog)
    # print('*' * 42)
    # print(input_prog)
    # print('*' * 42)
    ERROR_GRAMMAR = read_grammar(grammarFile)
    terminals = ERROR_GRAMMAR.get_alphabet()

    prog_tokens = get_token_list(input_prog, terminals)
    error_rules = predict_error_rules(grammarFile, modelsDir, gpuToUse, input_prog, True)
    actual_tokens = get_actual_token_list(input_prog, terminals)

    repaired_prog = repair(ERROR_GRAMMAR, max_cost, prog_tokens, error_rules, actual_tokens)
    if repaired_prog is not None:
        repaired_prog = repaired_prog.replace('_white_space_', ' ').replace('_NEWLINE_', '\n').replace("\\n", '\n')
    else:
        repaired_prog = input_prog

    # Here, you can implement code to retrieve the text from the provided URL
    # and process it as needed. For simplicity, we'll just return the decoded URL.

    fix = repaired_prog[:-3]
    print(fix)
    return fix


if __name__ == '__main__':
    app.run()
