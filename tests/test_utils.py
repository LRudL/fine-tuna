from dataclasses import dataclass
from finetuna.utils import dataclass_to_dict

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

