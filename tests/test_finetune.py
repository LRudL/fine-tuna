import os
import pytest
import tempfile
import random
import shutil

import finetuna
from finetuna.finetune import Finetuning, FTConfig
from finetuna.datagen.gen import DataGenerator, DataHolder, template_filler_fn, completion_maker_fn
from finetuna.completers import DummyCompleter
from finetuna.utils import write_to_jsonl

@pytest.fixture
def temp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def make_dummy_datagen(name, dir):
    dg = DataGenerator(
        lambda _ : "prompt",
        lambda _, __ : "completion",
        lambda : None,
        name = name
    )
    dg.generate(10)
    dg.save(custom_dir=dir)
    assert DataGenerator.name_exists(name, custom_dir=dir)
    return dg

class DummyFinetuning(Finetuning):
    def __init__(
        self, 
        datagen_name,
        ft_config,
        name,
        description,
        custom_dir,
        skip_exists_check=False,
        skip_save=False
    ):
        super().__init__(
            datagen_name,
            FTConfig("dummy_model"),
            name = name,
            description = "if you see this outside a test context, something has gone horribly wrong",
            custom_dir = custom_dir,
            skip_exists_check = skip_exists_check,
            skip_save=False
        )
    
    @staticmethod
    def load(name, custom_dir):
        return Finetuning.load(
            name,
            constructor=DummyFinetuning,
            custom_dir=custom_dir
        )
    
    def start(self):
        return 0
    
    def check(self):
        return 0
    
    def is_done(self):
        return False

def test_finetune_saving(temp_dir):
    dg = make_dummy_datagen("dummy_datagen", temp_dir)
    dg2 = make_dummy_datagen("dummy_datagen2", temp_dir)
    ft = DummyFinetuning("dummy_datagen", FTConfig("dummy_model"), "dummy_finetune", "", temp_dir)
    ft2 = DummyFinetuning("dummy_datagen2", FTConfig("dummy_model2"), "dummy_finetune2", "", temp_dir)
    ft.save()
    ft2.state.model_ptr = "agi"
    ft2.save()
    ft_loaded = DummyFinetuning.load(name="dummy_finetune", custom_dir=temp_dir)
    ft_loaded2 = DummyFinetuning.load(name="dummy_finetune2", custom_dir=temp_dir)
    assert ft_loaded.state.__dict__ == ft.state.__dict__
    assert ft_loaded.__dict__ == ft.__dict__
    assert isinstance(ft_loaded.state.ft_config, FTConfig)
    # now test that the other one can also be loaded:
    assert ft_loaded2.state.__dict__ == ft2.state.__dict__
    assert ft_loaded2.__dict__ == ft2.__dict__
    assert isinstance(ft_loaded2.state.ft_config, FTConfig)
