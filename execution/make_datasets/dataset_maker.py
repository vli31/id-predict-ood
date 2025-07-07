import sys 
sys.path.append('.')
import os
import random
import pandas as pd 
import numpy as np 
from enum import Enum
from utils.data import Heuristic, get_heuristics
from dyck_utils import all_opens, all_closes, get_close_of, DyckString
from tqdm import tqdm 

balancedHeuristic, matchedHeuristic = get_heuristics()

class StringGenerator(): 
    def __init__(self, heuristics:list[Heuristic], cdns:list[bool], n_brackets:int = 1): 
        self.heuristics = heuristics 
        self.cdns = cdns
        assert len(heuristics) == len(cdns), "heuristics and cdns must have the same length"
        assert type(cdns[0]) == bool, "cdn must be a boolean"

        self.CATALAN =  [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796, 58786, 208012, 742900, 2674440, 9694845, 35357670, 129644790, 477638700, 1767263190, 6564120420]
        # self.description describes what kind of strings are generated, e.g. balanced enclosed
        self.description = " ".join([h.positive if cdn else h.negative for h, cdn in zip(heuristics, cdns)])
        self.reps = 0
        self.get_ur_str = lambda len_str: "".join([random.choice(self.opens + self.closes) for _ in range(len_str)])

        self.opens = all_opens[:n_brackets]
        self.closes = all_closes[:n_brackets]

    def requires_heuristic(self, name:str): 
        for heuristic, cdn in zip(self.heuristics, self.cdns): 
            if (heuristic.name == name) and (cdn == True): 
                return True 
        return False
    
    def get_uniform_bal_str(self, length:int):
        assert length % 2 == 0, "length of balanced string must be even"
        
        s = DyckString()
        s.append_open(random.choice(self.opens))

        for i in range(1, length):
            r = s.elevation() # current elev
            k = length - i # number of characters to generate
            p_closed = r*(k+r+2)/(2*k*(r+1)) # probability of choosing an open_paren next
            # see https://dl.acm.org/doi/pdf/10.1145/357084.357091
            if random.random() < p_closed:
                s.append_close()
            else:
                s.append_open(random.choice(self.opens))
        return str(s)   

    def get_uniform_matched_str(self, length:int): 
        assert length % 2 == 0, "length of matched string must be even"
        open_half = [random.choice(self.opens) for _ in range(length//2)]
        close_half = [get_close_of(open_br) for open_br in open_half]
        bits = open_half + close_half
        random.shuffle(bits)
        gen_str = "".join(bits)
  
        return gen_str
    
    def get_unbalanced_enclosed_str(self, length: int):
        assert length >= 6, "length must be at least 6 to be unbalanced enclosed"
        while True:
            inner_str = self.get_ur_str(length - 2)
            gen_str = f"({inner_str})"
            if not balancedHeuristic.fn(gen_str):
                return gen_str
        
    def get_uniform_str(self, length:int): 
        self.reps += 1 
        if self.reps >= 100: 
            raise ValueError(f"Too many failures when attempting to generate a length {length} {self.description} string. Are you sure this is possible?")
        
        if not self.requires_heuristic("balanced") and self.requires_heuristic("enclosure"):
            if length < 6:
                return self.get_uniform_str(6)  # Recursively call with minimum length of 6

        if self.requires_heuristic("balanced"):
            gen_str = self.get_uniform_bal_str(length)

        elif self.requires_heuristic("matched"):
            gen_str = self.get_uniform_matched_str(length)
        
        elif (not self.requires_heuristic("balanced")) and self.requires_heuristic("enclosure"):
            gen_str = self.get_unbalanced_enclosed_str(length)

        else: 
            gen_str = self.get_ur_str(length)

        for heuristic, cdn in zip(self.heuristics, self.cdns):
            if heuristic.fn(gen_str) != cdn: 
                # try again
                return self.get_uniform_str(length) 
            
        # we have a string that satisfies all the constraints
        self.reps = 0
        return gen_str

class DistType(Enum):
    Uniform = "uniform"
    Binomial = "binomial"
    def __eq__(self, other):
        return self.value == other.value

def generate_uniform_random_string(string_generator: StringGenerator,
                                   length_dist:DistType, length_params:tuple):
    # aka sample_method = "uniform_random"
    if length_dist == DistType.Uniform: 
        a, b = length_params
        a = 1 if a // 2 == 0 else a // 2 
        b = b // 2
        str_len = 2*random.randint(a, b) # even number by length param 
    elif length_dist == DistType.Binomial: 
        n, p = length_params 
        str_len = 2 + 2*np.random.binomial(n/2-1, p) # even number between 2 and n

    return string_generator.get_uniform_str(str_len) 

def make_datasets(heuristics:list[Heuristic], cdns:list[bool], save_dir:str,
                  total_gen:int, length_dist:DistType, length_params:tuple, n_brackets:int = 1):
    assert len(heuristics) == len(cdns), "heuristics and cdns must have the same length"
    
    gen_set = set()
    string_generator = StringGenerator(heuristics, cdns, n_brackets=n_brackets)
    
    with tqdm(total=total_gen) as pbar:
        while len(gen_set) < total_gen:
            new_string = generate_uniform_random_string(string_generator, length_dist, length_params)
            if new_string not in gen_set:
                gen_set.add(new_string)
                pbar.update(1)
    
    df = pd.DataFrame({'string': list(gen_set)})

    save_path = []
    for i, heuristic in enumerate(heuristics):
        df[heuristic.name] = cdns[i]
        if cdns[i]: save_path.append((heuristic.name, heuristic.positive)) 
        else: save_path.append((heuristic.name, heuristic.negative))
    save_path.sort()

    save_path = save_dir + 'ur' + str(n_brackets) + '_' + length_dist.value + \
                (str(length_params).strip().replace(' ', '') if length_params != None else '') + \
                '_' + str(total_gen) + '_' + '_'.join([f'{name}' for _, name in save_path]) + '.csv'

    df.to_csv(save_path, index=False)

NUM_GENERATE = 500000
NUM_DYCK, NUM_VER = 5, 3

def get_dyck_data(heuristics:list[Heuristic], cdns:list[list[bool]]): 
    for num in range(NUM_DYCK): 
        n_brackets = num + 1         
        for i in range(NUM_VER):
            save_dir = f"datasets/dyck{n_brackets}/{'-'.join([heuristic.name[0:3] for heuristic in heuristics])}/{i+1}/baseline/" # bal-enc
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            print(save_dir)
            for dist, params in [( DistType.Uniform, (1,40)), #(DistType.Binomial, (40,0.25)), 
                                 (DistType.Binomial, (40,0.5)), (DistType.Binomial, (40,0.75))]:
                print(i, dist, params)
                for cdn_lst in cdns: 
                    make_datasets(heuristics, cdn_lst, save_dir, NUM_GENERATE, 
                                  dist, params, n_brackets=n_brackets)

# get_dyck_data([balancedHeuristic, enclosureHeuristic], [[True, True], [False, False], [False, True]])
# get_dyck_data([balancedHeuristic, matchedHeuristic], [[True, True], [False, False], [False, True]])
# get_dyck_data([matchedHeuristic, enclosureHeuristic], [[True, True], [False, False], [False, True], [True, False]])


# balanced, enclosed
def get_data(heuristics:list[tuple[Heuristic, Heuristic]]):
    data_info = []
    for h1, h2 in heuristics: 
        for num in range(NUM_DYCK): 
            n_brackets = num + 1 
            for i in range(NUM_VER):
                save_dir = f"datasets/dyck{n_brackets}/{'-'.join([heuristic.name[0:3] for heuristic in heuristics])}/{i+1}/baseline/" 
                for file in os.listdir(save_dir): 
                    info = file.replace('.csv', '').split("_")
                    data_info.append({"save_dir": save_dir, 
                                      "dyck": n_brackets, "ver": i+1, 
                                      "filename": file, 'data_dist': info[1],
                                      "size": info[2], h1: info[3], h2: info[4]})
    return pd.DataFrame(data_info)

# get_data([("balanced", "enclosure"), ("balanced", "matched"), ("matched", "enclosure")])
# empirical distribution of each 

def generate_ur_str(length_dist:DistType, length_params:tuple, n_brackets:int):
    get_ur_str = lambda len_str: "".join([random.choice(all_opens[:n_brackets] + all_closes[:n_brackets]) for _ in range(len_str)])
    # aka sample_method = "uniform_random"
    if length_dist == DistType.Uniform: 
        a, b = length_params
        a = 1 if a // 2 == 0 else a // 2 
        b = b // 2
        str_len = 2*random.randint(a, b) # even number by length param 
    elif length_dist == DistType.Binomial: 
        n, p = length_params 
        str_len = 2 + 2*np.random.binomial(n/2-1, p) # even number between 2 and n
    
    return get_ur_str(str_len)

def samp_strs(total_gen:int, length_dist:DistType, length_params:tuple, n_brackets:int, repeat:bool):
    bal_mat = 0 
    unbal_mat = 0 
    unbal_unmat = 0
    rpts = 0 

    gen_lst = []
    gen_set = set()
    with tqdm(total=total_gen) as pbar:
        while len(gen_lst if repeat else gen_set) < total_gen:
            new_string = generate_ur_str(length_dist, length_params, n_brackets)

            if new_string not in gen_set or repeat: 
                # if new_string not in gen_set:
                ifbal = balancedHeuristic.fn(new_string)
                ifmat = matchedHeuristic.fn(new_string) 
                
                if ifbal and ifmat: bal_mat += 1
                elif not ifbal and ifmat: unbal_mat += 1
                elif not ifbal and not ifmat: unbal_unmat += 1
                else: raise (ValueError("there was an error in categorizing the string"))
            
            if new_string in gen_set:
                rpts += 1 
            else: 
                gen_set.add(new_string)

            gen_lst.append(new_string)
            pbar.update(1)
    print("Repeats", rpts)
    return bal_mat, unbal_mat, unbal_unmat

print("bal_enc_mat", "unbal_enc_mat", "unbal_unenc_mat", "unbal_enc_unmat", "unbal_unenc_unmat")
for n_brackets in range(1, 6):
    print("dyck ", n_brackets)
    print("uniform(1,40)")
    print(samp_strs(500000, DistType.Uniform, (1, 40), n_brackets, True))
    print("binomial(40,0.5)")
    print(samp_strs(500000, DistType.Binomial, (40, 0.5), n_brackets, True))
    print("binomial(40,0.75)")
    print(samp_strs(500000, DistType.Binomial, (40, 0.75), n_brackets, True))
