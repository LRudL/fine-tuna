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

def random_fn_multiplexor(*fns, weights=None):
    """
    Returns a function that randomly selects from a list of functions.
    """
    if weights is None:
        weights = [1 / len(fns)] * len(fns)
    assert len(fns) == len(weights)
    def f(*args, **kwargs):
        return random.choices(fns, weights=weights)[0](*args, **kwargs)
    return f