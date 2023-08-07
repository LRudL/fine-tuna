import os
import json
from copy import deepcopy
import random

from datetime import datetime

def files_wo_suffix(path):
    """
    Returns all files in a directory without their suffixes.
    """
    return [os.path.splitext(f)[0] for f in os.listdir(path)]

def dict_key_filter(dict, keys):
    """
    Returns a new dictionary with only the specified keys.
    """
    return {key: dict[key] for key in keys}

def load_from_jsonl(filepath, only_retain_keys=None):
    """
    Load a JSONL file into a Python list.
    """
    with open(filepath, "r") as f:
        dataset = []
        for line in f:
            obj = json.loads(line)
            if only_retain_keys != None:
                obj = dict_key_filter(obj, only_retain_keys)
            dataset.append(obj)
    return dataset

def write_to_jsonl(dataset, filepath, only_keys=None):
    """
    Write a Python list to a JSONL file.
    """
    with open(filepath, "w") as f:
        for item in dataset:
            if only_keys != None:
                item = dict_key_filter(item, only_keys)
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

def timestr():
    now = datetime.now()
    formatted_date_time = now.strftime("%y-%m-%dT%H:%M")
    return formatted_date_time

def dataclass_to_dict(dc):
    """
    Converts a dataclass into a dictionary. This is done recursively:
    if a property of the dataclass links to another dataclass, that too
    is converted into a dictionary.
    """
    d = deepcopy(dc.__dict__)
    for key, value in d.items():
        if hasattr(value, "__dict__"):
            d[key] = dataclass_to_dict(value)
    return d

def copy_file(source_path, destination_path):
    with open(source_path, 'rb') as source_file:
        with open(destination_path, 'wb') as destination_file:
            # Read the content of the source file
            content = source_file.read()
            # Write the content to the destination file
            destination_file.write(content)

def dict_without_nones(dict):
    dict = deepcopy(dict)
    to_delete = []
    for key, value in dict.items():
        if value is None:
            to_delete.append(key)
    for key in to_delete:
        del dict[key]
    return dict

def duplicate_count_dict(l):
    """
    Returns a dictionary of the form {item: count} for a list of items.
    """
    d = {}
    for item in l:
        if item not in d.keys():
            d[item] = 0
        d[item] += 1
    return {
        key: value
        for key, value in d.items()
        if value > 1
    }