import os
import pytest
import tempfile
import random
import shutil
from copy import deepcopy

import finetuna
from finetuna.datagen.gen import DataGenerator, DataHolder, template_filler_fn, completion_maker_fn, get_openai_preprocess_hooks
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
    
    def test_counting(self):
        dg = DataGenerator(lambda _ : "", lambda _, __ : "", lambda : None)
        dg.add_item("", "", {1: 1})
        dg.add_item("", "", {1: 2})
        dg.add_item("", "", {1: 2})
        dg.add_item("", "", {1: 3})
        dg.add_item("", "", {1: 2})
        count = dg.count_by(1)
        assert count == {1: 1, 2: 3, 3: 1}
    
    def test_hook_simple(self):
        dg = DataGenerator(lambda _ : "", lambda _, __ : "", lambda : None)
        dg.add_item("abc", "xyz", {1: 1})
        dg.run_hook_for_point(lambda latent_state, prompt, completion,  : (latent_state, prompt, "xxx"), 0)
        assert dg.dataset[0]["completion"] == "xxx"
    
    def test_hooks(self):
        dg = DataGenerator(lambda _ : "", lambda _, __ : "", lambda : None)
        dg.add_item("abc", "def", {1: 1})
        dg.add_hook(lambda latent_state, prompt, completion : (latent_state, prompt, " " +  completion))
        dg.add_item("pqr", "stu", {1: 1})
        assert dg.dataset[0]["completion"] == " def"
        assert dg.dataset[1]["completion"] == " stu"
    
    def test_openai_preprocessor_hooks(self):
        dg = DataGenerator(lambda _ : "", lambda _, __ : "", lambda : None)
        dg.add_item("abc", "def", {1: 1})
        dg.add_item("pqr", "stu", {1: 1})
        dg.add_hook(get_openai_preprocess_hooks(
            prompt_end="BANANA",
            completion_end="APPLE"))
        dg.add_item("xyz", "123", {1: 1})
        for item in dg.dataset:
            assert item["prompt"].endswith("BANANA")
            assert item["completion"].endswith("APPLE")
        dataset = deepcopy(dg.dataset)
        dg.add_hook(get_openai_preprocess_hooks(
            prompt_end="BANANA",
            completion_end="APPLE"))
        assert dataset == dg.dataset, "Adding the same hook twice should not change the dataset."
    
    def test_count_by(self):
        dg = DataGenerator(lambda _ : "", lambda _, __ : "", lambda : None)
        dg.add_item("abc", "def", {"prop": 1})
        dg.add_item("pqr", "stu", {"prop": 2})
        dg.add_item("xyz", "123", None)
        assert dg.count_by("prop") == {1: 1, 2: 1}
        
        
        

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
    
    def test_DataHolder_from_file_list(self, temp_dir):
        dataset1 = self.get_dataset()
        dataset2 = self.get_dataset()
        path1 = f"{temp_dir}/__test_dataset1.jsonl"
        path2 = f"{temp_dir}/__test_dataset2.jsonl"
        write_to_jsonl(dataset1, path1)
        write_to_jsonl(dataset2, path2)
        dh = DataHolder([path1, path2])
        assert dh.dataset == dataset1 + dataset2
        assert dh.latent_states == [None for _ in range(200)]
    
    def test_DataHolder_from_file_list_with_latent_states(self, temp_dir):
        dataset1 = self.get_dataset()
        dataset2 = self.get_dataset()
        path1 = f"{temp_dir}/__test_dataset1.jsonl"
        path2 = f"{temp_dir}/__test_dataset2.jsonl"
        write_to_jsonl(dataset1, path1)
        write_to_jsonl(dataset2, path2)
        latent_states = [1 for _ in range(100)] + [2 for _ in range(100)]
        assert len(latent_states) == len(dataset1) + len(dataset2), "The test was written wrong"
        dh = DataHolder([path1, path2], latent_states)
        assert dh.dataset == dataset1 + dataset2
        assert dh.latent_states == latent_states
        
    def test_DataHolder_from_file_list_with_nested_latent_states(self, temp_dir):
        dataset1 = self.get_dataset()
        dataset2 = self.get_dataset()
        path1 = f"{temp_dir}/__test_dataset1.jsonl"
        path2 = f"{temp_dir}/__test_dataset2.jsonl"
        write_to_jsonl(dataset1, path1)
        write_to_jsonl(dataset2, path2)
        latent_states = [[1 for _ in range(100)]] + [[2 for _ in range(100)]]
        assert len(latent_states[0]) + len(latent_states[1]) == len(dataset1) + len(dataset2), "The test was written wrong"
        dh = DataHolder([path1, path2], latent_states)
        assert dh.dataset == dataset1 + dataset2
        assert dh.latent_states == latent_states[0] + latent_states[1]
    
    def test_DataHolder_from_mixed_list_with_nested_latent_states(self, temp_dir):
        dataset1 = self.get_dataset()
        dataset2 = self.get_dataset()
        path1 = f"{temp_dir}/__test_dataset1.jsonl"
        write_to_jsonl(dataset1, path1)
        latent_states = [[1 for _ in range(100)]] + [[2 for _ in range(100)]]
        dh = DataHolder([path1, dataset2], latent_states)
        assert dh.dataset == dataset1 + dataset2
        assert dh.latent_states == latent_states[0] + latent_states[1]
        dh = DataHolder([dataset2, path1], [latent_states[1], latent_states[0]])
        assert dh.dataset == dataset2 + dataset1
        assert dh.latent_states == latent_states[1] + latent_states[0]
    
    def test_DataHolder_merging(self):
        dataset1 = self.get_dataset()
        dataset2 = self.get_dataset()
        dh1 = DataHolder(dataset1)
        dh2 = DataHolder(dataset2)
        dh = dh1 + dh2
        assert len(dh.dataset) == 200
        assert dh.dataset[:100] == dataset1
        assert dh.dataset[100:] == dataset2
        assert dh.latent_states[:100] == dh1.latent_states
        assert dh.latent_states[100:] == dh2.latent_states
    
    def test_item_manipulation(self):
        dataset = [{"prompt": "abc", "completion": "def"}]
        dh = DataHolder(dataset)
        assert len(dh.dataset) == 1
        dh.add_item("pqr", "stu", {"prop": 1})
        dh.add_item("xyz", "123", {"prop": 2})
        dh.add_item("xyz", "345", {"prop": 2})
        assert len(dh.dataset) == 4
        assert len(dh.latent_states) == 4
        assert dh.latent_states[0] == None
        assert dh.latent_states[-1]["prop"] == 2
        subset = dh.subset([0, 3])
        assert len(subset.dataset) == 2
        assert subset.dataset[0]["prompt"] == "abc"
        assert subset.dataset[1]["prompt"] == "xyz"
        assert subset.latent_states[0] == None
        assert subset.latent_states[1]["prop"] == 2
        dh.delete_item(1)
        assert len(dh.dataset) == 3
        assert dh.dataset[0]["prompt"] == "abc"
        assert dh.dataset[1]["prompt"] == "xyz"
        dh.add_item("pqr", "stu", {"prop": 1})
        dh.add_item("pqr", "stu", {"prop": 3})
        dh_dict = dh.partition_by("prop")
        print(dh_dict)
        assert len(dh_dict[None]) == 1
        assert len(dh_dict[1]) == 1
        assert len(dh_dict[2]) == 2
        assert len(dh_dict[3]) == 1
        dh.remove_duplicate_prompts()
        assert len(dh) == 1
        assert dh.dataset[0]["prompt"] == "abc"
        
        
        
