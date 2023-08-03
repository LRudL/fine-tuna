import dill
from jinja2 import Environment, StrictUndefined, Undefined
import openai
import os
import random
from typing import Callable, Any, Union
from collections import Counter
from functools import reduce

from finetuna.utils import files_wo_suffix, load_from_jsonl, write_to_jsonl, random_fn_multiplexor
from finetuna.consts import OPENAI_API_KEY, DATASETS_PATH, DATA_GENERATORS_PATH
from finetuna.completers import Completer

openai.api_key = OPENAI_API_KEY

LatentState = Any

class DataHolder:
    """
    Useful for loading existing datsets that weren't generated
    by a DataGenerator.
    """
    def __init__(
        self, 
        jsonl_path_or_dataset,
        latent_states = None,
        name : Union[str, None] = None
    ):

        self.name : str = name if name is not None else "unnamed_dataset"
        self.hooks = []
        
        x = jsonl_path_or_dataset
        skip_latents_init = False
        initialised_by_name = False
        
        if isinstance(x, str):
            if os.path.exists(x):
                # Then it's a path to a JSONL file
                # Then it's a path to a JSONL file
                self.dataset = load_from_jsonl(jsonl_path_or_dataset)
            else:
                # Then it's a dataset name
                # (or someone messed up and is about to get an error)
                initialised_by_name = True
                if latent_states is None:
                    skip_latents_init = True
                self.name = x
                data_holder = DataHolder.load(x)
                self.__dict__.update(vars(data_holder))
        elif isinstance(x, list) or isinstance(x, tuple):
            # Then it's a list of data points, or a list of Union[dataset, path]
            if isinstance(x, tuple):
                x = list(x)
            if len(x) == 0 or (isinstance(x[0], dict) and "prompt" in x[0].keys()):
                # Then it's a list of data points
                for it in x:
                    assert "prompt" in it.keys() and "completion" in it.keys(), "jsonl_path_or_dataset should consist of {'prompt': ..., 'completion': ...} dicts if it is a list of data points"
                self.dataset = x 
            else:
                # Then it's a list of Union[dataset, path], try parsing
                # every list element as a DataHolder
                if latent_states != None and isinstance(latent_states[0], list):
                    # then we have nested latent_states lists
                    dataholders = [
                        DataHolder(d, latent_states=ls)
                        for d, ls in zip(x, latent_states)
                    ]
                else:
                    dataholders = [DataHolder(d) for d in x]
                dataholder = reduce(lambda x, y: x + y, dataholders)
                self.dataset = dataholder.dataset
                if latent_states != None and not isinstance(latent_states[0], list):
                    # then we don't have nested latent state lists,
                    # so they weren't passed into the dataholders,
                    # but we still want to overwrite them
                    self.latent_states = latent_states
                else:
                    # then we have nested latent state lists and passed them in,
                    # or we don't have any latent states to overwrite (so we just use the ones from the dataholder, a list of None)
                    self.latent_states = dataholder.latent_states
                skip_latents_init = True
        if not skip_latents_init:
            if latent_states is not None:
                assert isinstance(latent_states, list), "latent_states must be a list of latent states"
                assert len(latent_states) == len(self.dataset), "latent_states must be the same length as the dataset"
                self.latent_states = latent_states
            else:
                self.latent_states = [None] * len(self.dataset)
        
        assert hasattr(self, "dataset"), "dataset failed to be initialised"
        assert hasattr(self, "latent_states"), "latent_states failed to be initialised"
        assert hasattr(self, "name"), "name failed to be initialised"
        assert isinstance(self.name, str), "name must be a string"
        assert hasattr(self, "hooks"), "hooks failed to be initialised"
    
    def __add__(self, other):
        if not isinstance(other, DataHolder):
            raise NotImplementedError("Can only add DataHolders to other DataHolders.")
        return DataHolder(
            self.dataset + other.dataset,
            self.latent_states + other.latent_states,
            self.name
        )
    
    @staticmethod
    def name_exists(name : str, custom_dir = None) -> bool:
        dir = os.path.dirname(
            DataHolder.get_dataset_path("placeholder_name", dir=custom_dir)
        )
        if name in files_wo_suffix(dir):
            return True
        return False
    
    @staticmethod
    def create_or_load(
        name : str,
    ) -> 'DataHolder':
        if DataHolder.name_exists(name):
            return DataHolder.load(name)
        return DataHolder(
            name
        )
     
    @staticmethod
    def get_dataset_path(name, dir=None) -> str:
        path = f"{DATASETS_PATH}/{name}.jsonl"
        if dir is not None:
            path = f"{dir}/{path}"
        return path
    def dataset_path(self, dir=None) -> str:
        return DataHolder.get_dataset_path(self.name, dir=dir)
    
    @staticmethod
    def get_data_holder_path(name, dir=None) -> str:
        path = f"{DATA_GENERATORS_PATH}/{name}.pkl"
        if dir is not None:
            path = f"{dir}/{path}"
        return path
    def data_holder_path(self, dir=None) -> str:
        return DataHolder.get_data_holder_path(self.name, dir=dir)
    
    @staticmethod
    def load(name : str, dir=None) -> 'DataHolder':
        with open(DataHolder.get_data_holder_path(name, dir=dir), "rb") as f:
            data_holder = dill.load(f)
        
        # in case the class definition has changed, we load into a new copy:
        new_data_holder = DataHolder(
            data_holder.dataset,
            data_holder.latent_states,
            name = data_holder.name
        )
        for prop in data_holder.__dict__.keys():
            setattr(new_data_holder, prop, getattr(data_holder, prop))
        
        dataset = load_from_jsonl(DataHolder.get_dataset_path(name, dir=dir))
        new_data_holder.dataset = dataset
        # the above allows overriding the pickled file with manual .jsonl changes
        
        assert len(data_holder.dataset) == len(data_holder.latent_states), "Dataset and latent states must be the same length."
        return new_data_holder
    
    def save(self, warn_if_exists=False, custom_dir=None):
        if warn_if_exists:
            assert not DataHolder.name_exists(self.name), f"Dataset {self.name} already exists. Please choose a different name, or set warn_if_exists=True."
        # consts.py already makes these for the default case,
        # but not if a custom_dir is provided (useful for testing)
        data_holder_path = self.data_holder_path(custom_dir)
        dataset_path = self.dataset_path(custom_dir)
        os.makedirs(
            os.path.dirname(
                data_holder_path
            ), exist_ok=True
        )
        os.makedirs(
            os.path.dirname(
                dataset_path
            ), exist_ok=True
        )
        with open(data_holder_path, "wb") as f:
            print(self)
            dill.dump(self, f)
        self.save_data_to_jsonl(dataset_path)
        print(
            f"Wrote dataset {self.name} to {dataset_path}, and dataset object to {data_holder_path}. \nYou can load it with DataHolder.load('{self.name}{f', custom_dir={custom_dir}' if custom_dir is not None else ''}')."
        )
    
    def save_data_to_jsonl(self, filepath):
        """
        Save the generated dataset to a JSONL file.
        NOTE: in general, you should use the .save() method instead of this method.
        """
        write_to_jsonl(self.dataset, filepath)
    
    def print_sample(self, n=10):
        """
        Pretty-print a random size-n sample of the generated dataset.
        """
        n = min(n, len(self.dataset))
        bold = "\033[1m"
        reset = "\033[0m"
        boundary = "--------------------"
        print(boundary)
        indices = random.sample(range(len(self.dataset)), n)
        for i in indices:
            prompt_i = self.dataset[i]["prompt"]
            completion_i = self.dataset[i]["completion"]
            print(bold + prompt_i + reset + completion_i)
        print(boundary)
    
    def reset(self):
        self.dataset = []
        self.latent_states = []
    
    def add_item(
        self, 
        prompt, 
        completion, 
        latent_state = None,
        run_hooks = True
    ):
        """
        Adds an item. All hooks are run by default on the new point unless
        `run_hooks = False` is passed.
        """
        if run_hooks:
            for hook in self.hooks:
                (latent_state, prompt, completion) = hook(
                    latent_state,
                    prompt,
                    completion
                )
        self.dataset.append({
            "prompt": prompt,
            "completion": completion
        })
        self.latent_states.append(latent_state)
        assert len(self.dataset) == len(self.latent_states), "Error that really shouldn't happen in DataHolder.add_item: dataset and latent states must be the same length."
    
    def count_by(self, latent_state_prop):
        """
        Returns a dictionary of {latent_state_prop_value: count} for the given latent_state_prop.
        """
        vals = [
            ls[latent_state_prop]
            for ls in self.latent_states
            if ls is not None and latent_state_prop in ls.keys()
        ]
        return Counter(vals)
    
    def size(self):
        return len(self.dataset)
    
    def run_hook_for_point(self, hook_fn, i):
        """
        Runs all hooks for the given datapoint.
        """
        (latent_state, prompt, completion) = hook_fn(
            self.latent_states[i],
            self.dataset[i]["prompt"],
            self.dataset[i]["completion"]
        )
        self.latent_states[i] = latent_state
        self.dataset[i]["prompt"] = prompt
        self.dataset[i]["completion"] = completion
    
    def add_hook(self, fn, run_for_existing=True):
        """
        Add a hook (or list of hooks) that will be run after each datapoint is generated or added.
        Unless run_for_existing is set to False, the hook will also be run for all existing datapoints. 
        
        The hook should take three arguments: (latent_state, prompt, completion),
        and return a new triple of (latent_state, prompt, completion).
        
        We recommend hooks to be idempotent.
        """
        if isinstance(fn, list):
            for f in fn:
                self.add_hook(f, run_for_existing=run_for_existing)
            return
        if run_for_existing:
            for i in range(len(self.dataset)):
                self.run_hook_for_point(fn, i)
        self.hooks.append(fn)


class DataGenerator(DataHolder):
    """
    A DataHolder instance with functions for generating more points.
    The model behind data point generation is:
    1. a latent state generator (e.g. which type of prompt to generate,
    any information that both the prompt and completion need to share)
    2. a prompt generator function that takes in a latent state and 
    returns a prompt, where "prompt" means the same thing as it does
    in OpenAI finetuning data format
    3. a completion generator function that takes in the prompt and
    the latent state
    """
    def __init__(
        self,
        prompt_gen_fn : Callable[[LatentState], str],
        completion_gen_fn : Callable[[str, LatentState], str],
        latent_state_gen_fn : Callable[[], LatentState] = lambda: None,
        name : str = "unnamed_dataset"
    ):
        super().__init__([], [], name=name)
        self.prompt_gen_fn = prompt_gen_fn
        self.completion_gen_fn = completion_gen_fn
        self.latent_state_gen_fn = latent_state_gen_fn
   
    @staticmethod
    def load(name : str, dir=None) -> 'DataGenerator':
        with open(DataHolder.get_data_holder_path(name, dir=dir), "rb") as f:
            data_generator = dill.load(f)
        
        # in case the class definition has changed, we load into a new copy:
        new_data_generator = DataGenerator(
            data_generator.prompt_gen_fn,
            data_generator.completion_gen_fn,
            data_generator.latent_state_gen_fn,
            data_generator.name
        )
        for prop in data_generator.__dict__.keys():
            setattr(new_data_generator, prop, getattr(data_generator, prop))
        
        dataset = load_from_jsonl(DataHolder.get_dataset_path(name, dir=dir))
        new_data_generator.dataset = dataset
        # the above allows overriding the pickled file with manual .jsonl changes
        
        assert len(data_generator.dataset) == len(data_generator.latent_states), "Dataset and latent states must be the same length."
        return new_data_generator
   
    def generate(self, n=10):
        """
        Generate n samples of data.
        """
        for _ in range(n):
            latent_state = self.latent_state_gen_fn()
            self.latent_states.append(latent_state)
            prompt = self.prompt_gen_fn(latent_state)
            completion = self.completion_gen_fn(prompt, latent_state)
            self.dataset.append({
                "prompt": prompt,
                "completion": completion
            })
        return self.dataset[-n:]

    
def raise_not_implemented_error(s):
    raise NotImplementedError(s)


def template_filler_fn(
    template : str,
    use_strict_undefined = True,
    **variable_lists
) -> Union[Callable[[dict], str], Callable[[], str]]:
    """
    Returns a function that fills in a Jinja template string (i.e. {{variable}} is a variable) with random values from the given variable lists.
    
    The function returned by this function can take in a dictionary of special values to overwrite the random values. (e.g. when passing in LatentState with the above class)
    
    Example:
    >>> template = "My name is {{name}} and I am {{age}} years old."
    >>> name_list = ["Alice", "Bob", "Charlie"]
    >>> age_list = [18, 19, 20]
    >>> f = template_filler_fn(template, name=name_list, age=age_list)
    >>> f() -> "My name is Bob and I am 20 years old."
    """
    def random_template(special_vars = {}) -> str:
        # Create a Jinja2 environment with undefined variable handling.
        # Use Undefined to suppress errors for missing variables in lists.
        # (StrictUndefined will raise an error if a variable is missing in a list, otherwise it will just be blank.)
        env = Environment(
            undefined=StrictUndefined if use_strict_undefined else Undefined,
            autoescape=True)
        jinja_template = env.from_string(template)
        
        # Create a dictionary to store the random variable replacements.
        random_values = {
            var_name: random.choice(var_values) for var_name,
            var_values in variable_lists.items()
        }
        
        # overwrite with any special values:
        random_values.update(special_vars)
        
        # Render the template with the random values.
        return jinja_template.render(**random_values)
    return random_template

def completion_maker_fn(
    prompt_template : str,
    completer : Completer
):
    def get_completion(
        prompt_in_dataset : str,
        special_vars = {}
    ) -> str:
        """
        special_vars is for the latent state.
        `prompt_in_dataset` is called that to distinguish it from the prompt template used to construct the completion for the dataset
        """
        env = Environment(
            undefined=Undefined,
            autoescape=True)
        jinja_template = env.from_string(prompt_template)
         
        variables = {
            "prompt": prompt_in_dataset
        }
        
        if "prompt" in special_vars.keys():
            raise Exception("You cannot use 'prompt' as a latent state property name, because it is already used to store the prompt .")
        
        # overwrite with any special values:
        variables.update(special_vars)
        
        gpt_prompt = jinja_template.render(**variables)
        
        # now we get the completion:
        completion_text : str = completer(gpt_prompt)
        
        return completion_text
    return get_completion

def prompt_ends_hook_fn(end="\n\n###\n\n"):
    def fn(latent_state, prompt, completion):
        if not prompt.endswith(end):
            prompt += end
        return (latent_state, prompt, completion)
    return fn

def completion_starts_with_whitespace_hook(latent_state, prompt, completion):
    if not completion.startswith(" "):
        completion = " " + completion
    return (latent_state, prompt, completion) 

def completion_ends_hook_fn(end="\n"):
    def fn(latent_state, prompt, completion):
        if not completion.endswith(end):
            completion += end
        return (latent_state, prompt, completion)
    return fn

def get_openai_preprocess_hooks(prompt_end="\n\n###\n\n", completion_end="\n"):
    return [
        prompt_ends_hook_fn(prompt_end),
        completion_starts_with_whitespace_hook,
        completion_ends_hook_fn(completion_end)
    ]