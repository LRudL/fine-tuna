import os
import pytest
import tempfile
import random
import shutil

import finetuna
from finetuna.datagen.gen import DataGenerator, DataHolder, template_filler_fn, completion_maker_fn
from finetuna.completers import DummyCompleter
from finetuna.utils import write_to_jsonl

finetuna.completers.BLOCK_API_CALLS = True

class Test_template_filler_fn:
    def test_template_filler_fn(self):
        template = "My name is {{name}} and I am {{age}} years old."
        name_list = ["Bob the AGI"]
        age_list = [1]
        f = template_filler_fn(template, name=name_list, age=age_list)
        assert f() == "My name is Bob the AGI and I am 1 years old." # type: ignore
    
    def test_template_filler_fn_overwrite(self):
        template = "My name is {{name}} and I am {{age}} years old."
        name_list = ["Bob the AGI"]
        age_list = [1]
        f = template_filler_fn(template, name=name_list, age=age_list)
        assert f({"age": 2}) == "My name is Bob the AGI and I am 2 years old." # type: ignore

class Test_completion_maker_fn:
    def test_completion_maker_fn(self):
        completer = DummyCompleter(
            lambda filled_in_prompt : f"Echoing: {filled_in_prompt}."
        )
        prompt_template = "Hello, you {{adjective}} said {{prompt}}"
        f = completion_maker_fn(prompt_template, completer)
        assert f("hi", {"adjective": "cheerfully"}) == "Echoing: Hello, you cheerfully said hi." # type: ignore
        

@pytest.fixture
def temp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

class Test_DataGenerator:
    def construct_data_generator(self):
        dg = DataGenerator(
            lambda x: f"This is a prompt {x}.",
            lambda prompt, x: f"The value of 2x is {2 * x}",
            lambda: random.randint(0, 100),
            "__test_dataset"
        )
        return dg
        
    def test_DataGenerator(self):
        dg = self.construct_data_generator()
        dg.generate(100)
        assert len(dg.dataset) == 100
        assert dg.dataset[0].keys() == {"prompt", "completion"}
    
    def test_saving_and_loading(self, temp_dir):
        dg = self.construct_data_generator()
        dataset = dg.dataset
        dg.generate(100)
        dg.save(custom_dir=temp_dir)
        assert os.path.exists(dg.dataset_path(dir=temp_dir))
        dg_loaded = DataGenerator.load(dg.name, dir=temp_dir)
        assert dataset == dg_loaded.dataset
        assert dg_loaded.prompt_gen_fn(0) == dg.prompt_gen_fn(0)

class Test_DataHolder:
    def get_dataset(self):
        dg = DataGenerator(
            lambda x: f"This is a prompt {x}.",
            lambda prompt, x : f"The value of 2x is {2 * x}",
            lambda: random.randint(0, 100)
        )

        dg.generate(100)
        
        return dg.dataset
    
    def test_DataHolder_from_dataset(self):
        dataset = self.get_dataset()
        dh = DataHolder(dataset)
        assert dh.dataset == dataset
    
    def test_DataHolder_from_file(self, temp_dir):
        dataset = self.get_dataset()
        path = f"{temp_dir}/__test_dataset.jsonl"
        write_to_jsonl(dataset, path)
        dh = DataHolder(path)
        assert dh.dataset == dataset
        
