import argparse
import random
import time
import traceback
import openai
import threading
import tqdm
import datasets
from concurrent.futures import ThreadPoolExecutor

def start_session(i: int, url: str, model: str, dataset, stream: bool=False, max_tokens: int=2048) -> str:
  try:
    openai.api_key = "EMPTY" # Not support yet
    openai.api_base = url

    prompt = dataset[i]
    # create a chat completion
    response = openai.Completion.create(
      model=model,
      prompt=prompt,
      stream=stream,
      max_tokens=max_tokens,
      temperature=0.9,
    )
    # print the completion
    if stream:
      out = ""
      for chunk in response:
        out += str(chunk)
    else:
      out = response.choices[0].text
      total_tokens = response.usage.total_tokens
      completion_tokens = response.usage.completion_tokens
  except Exception as e:
    traceback.print_exc()
    return "", 0, 0
  
  return out, total_tokens, completion_tokens

def get_prompt(raw_dataset):
    dataset = []
    for conversations in raw_dataset["conversations"]:
        messages = []
        for data in conversations:
            out_data = {"role": "system", "content": ""}
            if data["user"] == "human":
                out_data["role"] = "user"
            if data["user"] == "gpt":
                out_data["role"] = "assitant"
            
            out_data["content"] = data["text"]
            messages.append(out_data)

        if messages[-1]["role"] == "gpt":
            messages = messages[:-1]
        
        prompt = "\n###".join([msg["role"] + ": " + msg["content"] for msg in messages]) + "\n### assistant: "
        if len(prompt) > 2048:
           continue
        dataset.append(prompt)
    return dataset

def main(args):
  dataset = datasets.load_dataset("theblackcat102/sharegpt-english", split="train")
  dataset = get_prompt(dataset)

  tik = time.time()
  tasks = []
  with ThreadPoolExecutor(max_workers=args.n_thread) as t:
    for i in range(args.total_task):
      task = t.submit(start_session, i=i, url=args.url, model=args.model_name, dataset=dataset, stream=False, max_tokens=args.max_tokens)
      tasks.append(task)
    
    results = []
    for task in tqdm.tqdm(tasks):
      results.append(task.result())

  n_tokens = sum([ret[2] for ret in results])
  n_queries = sum([1 for ret in results if ret[2] != 0])
  time_seconds = time.time() - tik
  print(
      f"Successful number: {n_queries} / {args.total_task}. "
      f"Time (Completion): {time_seconds}, n threads: {args.n_thread}, "
      f"throughput: {n_tokens / time_seconds} tokens/s."
      f"QPS: {n_queries / time_seconds} queries/s."
  )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model-name", type=str, default="vicuna")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--total-task", type=int, default=64)
    parser.add_argument("--n-thread", type=int, default=4)
    args = parser.parse_args()

    main(args)