import os

def files_wo_suffix(path):
    """
    Returns all files in a directory without their suffixes.
    """
    return [os.path.splitext(f)[0] for f in os.listdir(path)]