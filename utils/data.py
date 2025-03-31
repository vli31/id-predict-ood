import torch as t
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Callable

class Heuristic:
  def __init__(self, heuristic: str, description: str, 
               positive: str, negative: str, distinguishing_function: Callable[[str], bool]):
    self.name = heuristic # name of the heuristic
    
    self.description = description
    self.fn = distinguishing_function # returns True if string is a positive example of the heuristic, False if negative
    self.positive = positive # name of the positive category
    self.negative = negative # name of the negative category

  def __str__(self):
    return self.name
  
  def __repr__(self):
    return str(self)
  
  def __eq__(self, other):
    return self.name == other.name and self.description == other.description

def is_balanced_for_loop(parens: str) -> bool:
    '''
    Return True if the parens are balanced.
    Parens are just the ( and ) characters, no begin or end tokens.
    '''
    cumsum = 0
    for paren in parens:
        cumsum += 1 if paren == "(" else -1
        if cumsum < 0:
            return False
    
    return cumsum == 0

balancedHeuristic = Heuristic("balanced", 
                              "standard balanced/unbalanced heuristic",
                              "balanced", "unbalanced",
                              is_balanced_for_loop)
matchedHeuristic  = Heuristic("matched", 
                              "number of open brackets equals number of closed brackets",
                              "matched", "unmatched",
                              lambda s: s.count("(") == s.count(")"))
enclosureHeuristic = Heuristic("enclosure", 
                               "first and last brackets are ( and ) respectively",
                               "(x)", "(x(, )x), )x(",
                               lambda s: s[0] == "(" and s[-1] == ")")


class SimpleTokenizer:
    """A simple tokenizer that maps characters to integers. Borrowed from the Redwood Research Mlab curriculum."""

    START_TOKEN = 0
    PAD_TOKEN = 1
    END_TOKEN = 2
    base_d = {"[start]": START_TOKEN, "[pad]": PAD_TOKEN, "[end]": END_TOKEN}

    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        # the 3 is because there are 3 special tokens (defined just above)
        self.t_to_i = {**{c: i + 3 for i, c in enumerate(alphabet)}, **self.base_d}
        self.i_to_t = {i: c for c, i in self.t_to_i.items()}

    def tokenize(self, strs: List[str], max_len = None):
        def c_to_int(c: str) -> int:
            if c in self.t_to_i:
                return self.t_to_i[c]
            else:
                raise ValueError(c)

        if isinstance(strs, str):
            strs = [strs]

        if max_len is None:
            max_len = max((max(len(s) for s in strs), 1))

        ints = [
            [self.START_TOKEN] + [c_to_int(c) for c in s] + [self.END_TOKEN] + [self.PAD_TOKEN] * (max_len - len(s)) 
            for s in strs
        ]
        return t.tensor(ints)

    def decode(self, tokens) -> List[str]:
        assert tokens.ndim >= 2, "Need to have a batch dimension"
        def int_to_c(c: int) -> str:
            if c < len(self.i_to_t):
                return self.i_to_t[c]
            else:
                raise ValueError(c)

        return [
            "".join(int_to_c(i.item()) for i in seq[1:] if i != self.PAD_TOKEN and i != self.END_TOKEN)
            for seq in tokens
        ]

    def __repr__(self) -> str:
        return f"SimpleTokenizer({self.alphabet!r})"
        

class BracketsDataset(Dataset):
    """A dataset containing dataframes of strings and their labels according to various heuristics."""
    
    def __init__(self, data:dict|pd.DataFrame, alphabet:str="()", max_len:int=40,
                 dataset_info={}, name=None, ylabels_heuristic=balancedHeuristic):
        """
        data: dict or pd.Dataframe contain strings and labels according to various heuristics
        max_len: int, the maximum length of the tokenized sequences
        ylabels_heuristic: Heuristic to use as the target labels
        dataset_info: dict of additional information about the dataset
        """
        self.name = name
        self.dataset_info = dataset_info

        if type(data) == dict:
            self.df = pd.DataFrame.from_dict(data)
        elif type(data) == pd.DataFrame:
            self.df = data
        else: 
            raise(ValueError("dataset type not supported"))

        # x labels, aka strings constructed from the given alphabet
        self.strs = list(self.df["string"].values)

        # tokenized strings
        self.tokenizer = SimpleTokenizer(alphabet)
        self.toks = self.tokenizer.tokenize(self.strs, max_len=max_len)
        self.max_len = max_len

        # y labels
        self.ylabels_heuristic = ylabels_heuristic
        self.ylabels = t.tensor(self.df[ylabels_heuristic.name].values).bool()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx:slice|int):
        if isinstance(idx, slice):
            return self.__class__(self.df[idx], alphabet=self.tokenizer.alphabet, 
                                  max_len=self.max_len,
                                  ylabels_heuristic=self.ylabels_heuristic,
                                  dataset_info=self.dataset_info)
        elif isinstance(idx, int):
            input_string = self.strs[idx]
            input_tensor = self.toks[idx]
            target = self.ylabels[idx]
            return input_string, input_tensor, target

    def to(self, device) -> "BracketsDataset":
        self.toks = self.toks.to(device)
        self.ylabels = self.ylabels.to(device)
        return self

    @property
    def seq_length(self) -> int:
        return self.toks.size(-1)
    
    def set_ground_truth(self, ylabels_heuristic):
        # change the ground truth labels based on the heuristic
        # e.g. if the heuristic is "balanced", then the ground truth labels are the "balanced" column
        self.ylabels_heuristic = ylabels_heuristic
        self.ylabels = t.tensor(self.df[ylabels_heuristic.name].values).bool()

    def shuffle(self, data_shuffle_seed:int|None):
        if data_shuffle_seed is None: 
            return self 
        else: 
            new_dataset_info = self.dataset_info.copy()
            new_dataset_info["data_shuffle_seed"] = data_shuffle_seed
            return self.__class__(self.df.sample(n=len(self), random_state=data_shuffle_seed), 
                                alphabet=self.tokenizer.alphabet, max_len=self.max_len,
                                ylabels_heuristic=self.ylabels_heuristic, 
                                dataset_info=new_dataset_info)

