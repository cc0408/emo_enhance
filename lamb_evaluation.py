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
    backoff_time = 1
    while retry_cnt <= retries:
        try:
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
                backoff_time *= 1.5
            retry_cnt -= 1
    
    data = {
            "prompt": messages,
            "response": response,
            "created_at": str(datetime.now()),
        }
    return data



if __name__ == '__main__':
    request_batch_size = 20
    input_path = '/home/xuxi/emo_enhance/davinci3_kaggle.csv'
    input_file = pd.read_csv(input_path, encoding='latin-1')
    # print(input_file)
    original = input_file['original']
    lamb = input_file['lamb']
    label = input_file['label']
    output_path = '/home/xuxi/emo_enhance/eval/evaluation_davinci3_kaggle.csv'
    try:
        output_file = pd.read_csv(input_path, encoding='latin-1')
        output_file['is_better'] = [''] * 60
    except:
        input_file.to_csv(output_path)
        output_file = pd.read_csv(output_path)
    start=1
    # print(output_file['is_better'])
    for start in range(len(output_file)):
    
        if output_file.iloc[start]['is_better'] != '':
            continue
        else:
            break
    print(f"Starting to summarize from line {start}")
    
    progress_bar = tqdm.tqdm(total=len(range(start, len(output_file))))
    for idx in range(start, len(output_file)):
        messages = [{"role": "system", "content": "You will be given a sequence of text, the emotion label and two responses. The format of the input is: \n [Original text] \n The original text \n [Label] The emotion label of the original text \n [Response 1] \n The response generated by the first method \n [Response 2] \n The response generated by the second method \n. Please rate the strength of the emotion of these two responses by giving a integer score from 1 to 5, and return the scores for these two responses. The output format is: \n score1 (score for Response 1), score2 (score for Response 2)"}, 
                    {"role": "user", "content": f"[Original text] \n {output_file.iloc[idx]['original']} \n [Label] \n {output_file.iloc[idx]['label']} \n [Response 1] \n {output_file.iloc[idx]['davinci3']} \n [Response 2] \n {output_file.iloc[idx]['lamb']}"}]
        results = make_requests(
            engine="gpt-3.5-turbo-0613",
            messages=messages,
            retries=1,
            temperature=0.7,
            top_p=0.95,
            max_tokens=150,
            frequency_penalty=0,
            presence_penalty=0,
            stop_sequences=None,
            logprobs=1,
            n=1,
            best_of=1)
        # print(results)
        try:
            output_file['is_better'].iloc[idx] = results['response']['choices'][0]['message']['content']
        except:
            output_file['is_better'].iloc[idx] = 'failed'
        # assert False
        output_file.to_csv(output_path, index=False)
        progress_bar.update(1)