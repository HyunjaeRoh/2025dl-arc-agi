import json
import os
from typing import List, Dict, Generator, Optional
import random

message_templates = {
    "system_prompt": \
        '''You are a puzzle solving wizard. You are given a puzzle from the abstraction and reasoning corpus''',

    "user_message_template1": \
        '''Here are the example input and output pairs from which you should learn the underlying rule to later predict the output for the given test input:
        ----------------------------------------''',

    "user_message_template2": \
        '''----------------------------------------
        Now, solve the following puzzle based on its input grid by applying the rules you have learned from the training data.:
        ----------------------------------------''',

    "user_message_template3": \
        '''----------------------------------------
        What is the output grid? Only provide the output grid in the form as in the example input and output pairs. Do not provide any additional information:''',

    "user_message_template4": \
        '''----------------------------------------
        Considering the examples, the test input, and potentially the predicted output grid provided below, please describe the transformation rule you observed in a single, concise English sentence.
        Rule: '''
}

def load_data_from_a_task(
        all_examples,
        num_examples,
        num_test,
):
    datapoints = []
    window_size = num_examples + 1

    shuffled_examples = list(all_examples)
    random.shuffle(shuffled_examples)

    if len(shuffled_examples) < window_size:
        return []

    data_iter = (
        shuffled_examples[i * window_size : (i+1) * window_size]
        for i in range(len(shuffled_examples) // window_size)
    )

    for cnt, window in enumerate(data_iter):
        if cnt >= num_test:
            break

        train_part = window[:num_examples]
        test_part = window[num_examples]

        datapoint = {
            'examples': train_part,
            'test': test_part,
        }


    return None