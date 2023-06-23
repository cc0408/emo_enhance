import os
import json
import numpy as np
import random
import tqdm
import re
import argparse
import pandas as pd
from collections import OrderedDict
import os
import openai
# openai.api_type = "azure"
# openai.api_base = "https://flaml-east.openai.azure.com/"
# openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")
import pandas as pd
from datetime import datetime
import time
from datasets import load_dataset
random.seed(42)

def make_requests(
        engine, messages, max_tokens, temperature, top_p, 
        frequency_penalty, presence_penalty, stop_sequences, logprobs, n, best_of, retries=3, api_key=None, organization=None
    ):
    response = None
    target_length = max_tokens
    if api_key is not None:
        openai.api_key = api_key
    if organization is not None:
        openai.organization = organization
    retry_cnt = 0
    backoff_time = 2
    while retry_cnt <= retries:
        try:
            # print(messages)
            response = openai.ChatCompletion.create(
                model=engine,
                messages=messages,
                max_tokens=target_length,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop_sequences,
                n=n,
            )
            break
        except openai.error.OpenAIError as e:
            print(f"OpenAIError: {e}.")
            if "Please reduce your prompt" in str(e):
                target_length = int(target_length * 0.8)
                print(f"Reducing target length to {target_length}, retrying...")
            else:
                print(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                #backoff_time *= 2
            retry_cnt -= 1
    
    data = {
            "prompt": messages,
            "response": response,
            "created_at": str(datetime.now()),
        }
    return data
def make_comp(
        engine, messages, max_tokens, temperature, top_p, 
        frequency_penalty, presence_penalty, stop_sequences, logprobs, n, best_of, retries=3, api_key=None, organization=None
    ):
    response = None
    target_length = max_tokens
    if api_key is not None:
        openai.api_key = api_key
    if organization is not None:
        openai.organization = organization
    retry_cnt = 0
    backoff_time = 2
    while retry_cnt <= retries:
        try:
            # print(messages)
            response = openai.Completion.create(
                engine=engine,
                prompt=messages,
                max_tokens=target_length,
            )
            break
        except openai.error.OpenAIError as e:
            print(f"OpenAIError: {e}.")
            if "Please reduce your prompt" in str(e):
                target_length = int(target_length * 0.8)
                print(f"Reducing target length to {target_length}, retrying...")
            else:
                print(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                #backoff_time *= 2
            retry_cnt -= 1
    
    data = {
            "prompt": messages,
            "response": response,
            "created_at": str(datetime.now()),
        }
    return data

if __name__ == '__main__':
    int2label = {
        4: "sadness",
        2: "joy",
        0: "anger",
        1: "fear",
        3: "love",
        5: "surprise"
    }
    gemo = {
        0 : "anger",
        1 : "disgust",
        2 : "fear",
        3 : "joy",
        4 : "sadness",
        5 : "surprise"
    }
    dataset = load_dataset("csv",data_dir="/home/xuxi/emo_enhance/data/",
                            data_files={'train':'train.csv', 'test':'test.csv'}, 
                            column_names=["sentence", "label"])
    lamb = load_dataset("csv",data_dir="/home/xuxi/emo_enhance",
                            data_files={'test':'lamb_500.csv'}, 
                            column_names=["id", "ori", "la"])
    dataset = dataset.shuffle(seed=0)
    output_path = '/home/xuxi/emo_enhance/kaggle_v3.csv'
    res = [['label','original','lamb','gpt', 'davinci3']]
    dnum = {}
    for idx in range(0, 5):
        sentence = dataset['test'][idx]['sentence']
        label = dataset['test'][idx]['label']
        la = lamb['test'][idx+1]['la']
        label = int2label[label]
        tmp = dnum.get(label, 0)
        if tmp>9:
            continue
        dnum[label]=tmp+1
        messages = [{"role": "system", "content": "You are an emotional booster for editing the text by changing no more than 3 words and without punctuation so that the two sentences are mostly the same. Changes should be as small as possible to go undetected."},
                    {"role": "user","content": f"The original sentence is :{sentence}. The sentence that has a stronger {label} emotion and maintains the same semantics by adding and replacing within three words is:"}]
        results = make_requests(
            engine="gpt-3.5-turbo-0613",
            messages=messages,
            temperature=1,
            top_p=1,
            max_tokens=150,
            frequency_penalty=0,
            presence_penalty=0,
            stop_sequences=None,
            logprobs=1,
            n=1,
            best_of=1
        )
        gpt = results['response']['choices'][0]['message']['content'].strip('"')

        messages = f"You are an emotional booster for editing the text by changing no more than 3 words so that the two sentences are mostly the same.  Changes should be as small as possible to go undetected. The original sentence is :{sentence}. The sentence that has a stronger {label} emotion and maintains the same semantics by adding and replacing within three words is:"
        results = make_comp(
            engine="text-davinci-003",
            messages=messages,
            temperature=1,
            top_p=1,
            max_tokens=50,
            frequency_penalty=0,
            presence_penalty=0,
            stop_sequences=None,
            logprobs=1,
            n=1,
            best_of=1
        )
        da3 = results['response'].choices[0].text.strip()

        # print(sentence, gpt, '',sep='\n')
        res.append([label, sentence, la, gpt, da3])
        print(idx, end=' ',flush=True)
        if idx % 30 == 0:
            print('')
    res = pd.DataFrame(res)
    res.to_csv(output_path, index=False, header=False)
