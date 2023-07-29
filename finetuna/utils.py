import os
import json

def files_wo_suffix(path):
    """
    Returns all files in a directory without their suffixes.
    """
    return [os.path.splitext(f)[0] for f in os.listdir(path)]

def load_from_jsonl(filepath):
    """
    Load a JSONL file into a Python list.
    """
    with open(filepath, "r") as f:
        dataset = []
        for line in f:
            dataset.append(json.loads(line))
    return dataset

def write_to_jsonl(dataset, filepath):
    """
    Write a Python list to a JSONL file.
    """
    with open(filepath, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
