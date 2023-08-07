import pytest
import tempfile
import shutil
from dataclasses import dataclass
from finetuna.utils import dataclass_to_dict, load_from_jsonl, write_to_jsonl

@dataclass
class DummyDataclass:
    x : str
    y : str

@dataclass
class BigDummyClass:
    z : str
    a : DummyDataclass

def test_dataclass_to_dict():
    dc = DummyDataclass("x", "y")
    bdc = BigDummyClass("z", dc)
    assert dataclass_to_dict(bdc) == {"z": "z", "a": {"x": "x", "y": "y"}}

def test_dataclass_to_dict_non_mutating():
    dc = DummyDataclass("x", "y")
    bdc = BigDummyClass("z", dc)
    dict = dataclass_to_dict(bdc)
    dict["z"] = "zz"
    dict["a"]["x"] = "xx"
    assert bdc.z == "z"
    assert bdc.a.x == "x"

@pytest.fixture
def temp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_load_from_jsonl(temp_dir):
    test_data = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob", "extra": "blah blah blah"},
        {"id": 3, "name": "Charlie", "extra": "blah blah blah", "extra2": "blah blah blah"}
    ]
    test_file = f"{temp_dir}/test.jsonl"
    write_to_jsonl(test_data, test_file)
    dataset = load_from_jsonl(test_file, only_retain_keys=["name", "id"])
    assert dataset[0]["name"] == "Alice"
    assert dataset[1]["name"] == "Bob"
    assert dataset[2]["name"] == "Charlie"
    assert len(dataset) == 3
    assert len(dataset[0].keys()) == 2
    assert len(dataset[1].keys()) == 2
    assert len(dataset[2].keys()) == 2

