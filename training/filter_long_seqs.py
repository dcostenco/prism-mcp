import json
import os
from transformers import AutoTokenizer

def filter_file(input_path, output_path, max_len=2000):
    if not os.path.exists(input_path):
        return
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/xLAM-2-32b-fc-r")
    kept = 0
    dropped = 0
    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            data = json.loads(line)
            text_content = ""
            if "messages" in data:
                text_content = " ".join([m.get("content", "") for m in data["messages"]])
            elif "text" in data:
                text_content = data["text"]
            elif "prompt" in data:
                text_content = str(data["prompt"])
            
            if len(str(data)) / 3.5 > max_len:
                tokens = len(tokenizer.encode(text_content))
                if tokens > max_len:
                    dropped += 1
                    continue
            f_out.write(line)
            kept += 1
    print(f"{input_path}: Kept {kept}, Dropped {dropped}")

data_in = os.environ.get("PRISM_DATA_DIR", "/Users/admin/prism/training/data/combined")
data_out = os.environ.get("PRISM_DATA_OUT_DIR", "/Users/admin/prism/training/data/filtered")
os.makedirs(data_out, exist_ok=True)
filter_file(os.path.join(data_in, "train.jsonl"), os.path.join(data_out, "train.jsonl"))
filter_file(os.path.join(data_in, "valid.jsonl"), os.path.join(data_out, "valid.jsonl"))
