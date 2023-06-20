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
    backoff_time = 30
    while retry_cnt <= retries:
        try:
            print(messages)
            response = openai.ChatCompletion.create(
                engine=engine,
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
            retry_cnt += 1
    
    data = {
            "prompt": messages,
            "response": response,
            "created_at": str(datetime.now()),
        }
    return data



if __name__ == '__main__':
    print('key', openai.api_key)
    sentence = 'You are a good girl and I like you'
    emo = 'love'
    messages = [{"role": "system", "content": "Emotional Booster for Your Plain Text."}, 
                {"role": "user","content": f"The original sentence is :{sentence}. The sentence with stronger {emo} emotion and no change in semantics by adding words and changing words is:"}]
    results = make_requests(
        engine="gpt-35-turbo",
        messages=messages,
        temperature=1,
        top_p=1,
        max_tokens=150,
        frequency_penalty=0,
        presence_penalty=0,
        stop_sequences=None,
        logprobs=1,
        n=1,
        best_of=1)
    print(results)
    
    # request_batch_size = 20
    # input_path = '/Users/chanyh/data/EC_number.csv'
    # input_file = pd.read_csv(input_path, header=None)
    # descriptions = input_file.iloc[:, 1].to_list()
    # input_file['summary'] = ''
    # parsed = []
    # output_path = '/Users/chanyh/data/EC_number_summary.csv'
    # try:
    #     output_file = pd.read_csv(output_path)
    # except:
    #     input_file.to_csv(output_path)
    #     output_file = pd.read_csv(output_path)

    # output_file.iloc[:, 3] = output_file.iloc[:, 3].astype(str)
    # for start in range(len(output_file)):
    #     print(output_file.iloc[start, 3])
        
    #     if output_file.iloc[start, 3] != 'nan' :
    #         continue
    #     else:
    #         break
    
    # print(f"Starting to summarize from line {start}")
    # parsed = descriptions[:start]

    # progress_bar = tqdm.tqdm(total=len(range(start, len(output_file))))
    # for idx in range(start, len(output_file)):
    #     batch = descriptions[idx]
    #     messages = [{"role": "system", "content": "You are an expert on enzyme function."}, 
    #                 {"role": "user","content": f"You are explaining the function of an enzyme. Please extract the following enzyme function in less than 50 words. When extracting, please keep the biological terms: {batch}"}]
    #     sentence = 'You are a good girl and I like you'
    #     emo = 'love'
    #     messages = [{"role": "user","content": f"The original sentence is :{sentence}. The sentence with stronger {emo} emotion and no change in semantics by adding words and changing words is:"}]
    #     results = make_requests(
    #         engine="gpt-35-turbo",
    #         messages=messages,
    #         temperature=1,
    #         top_p=1,
    #         max_tokens=150,
    #         frequency_penalty=0,
    #         presence_penalty=0,
    #         stop_sequences=None,
    #         logprobs=1,
    #         n=1,
    #         best_of=1)
    #     # print(results)
    #     output_file['summary'].iloc[idx] = results['response']['choices'][0]['message']['content']
    #     # assert False
    #     output_file.to_csv(output_path, index=False)
    #     progress_bar.update(1)