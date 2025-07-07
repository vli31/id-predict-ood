"""
Training with wandb logs 
Last Updated: 8/18
"""

import yaml, os, argparse
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
import wandb
import random 
import numpy as np 
import pandas as pd 
import torch as t 
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from utils.data import BracketsDataset
from utils.model import get_transformer, get_LSTM, GPT, LSTMClassifier, N_CLASSES, ModelType

LOGS_DIR = "./logs/"
CUTOFF = 0.9

class Trainer:
    def __init__(self, model_type, num_repeats: int, save_csv: str, sweep: bool, train_data, indist_test_data, ood_test_data):
        assert t.cuda.is_available(), "!!!CUDA not available!!!"

        self.run = wandb.init(dir=LOGS_DIR)
        if sweep:
            self.config = wandb.config  # model config determined by sweep
        else:
            if model_type == ModelType.LSTM:
                self.config = yaml.safe_load(open(LSTM_CFG, "r"))
            elif model_type == ModelType.Transformer:
                self.config = yaml.safe_load(open(TRANSFORMER_CFG, "r"))

        # Initialize datasets
        self.len_train = len(train_data)
        self.train_data = self._repeat_and_shuffle_data(train_data, num_repeats, self.config["shuffle_seed"])
        self.indist_test_data = indist_test_data
        self.ood_test_data = ood_test_data

        # set random seed
        if self.config["rdm_seed"] is not None:
            self._set_seed(self.config["rdm_seed"])

        self.g = t.Generator()
        self.g.manual_seed(self.config["rdm_seed"] if self.config["rdm_seed"] is not None else 0)
        self.batch_size = self.config["batch_size"]
        self.loss_fn = t.nn.CrossEntropyLoss().cuda()

        # instantiate models
        self.model = self._initialize_model(model_type)

        self._get_optimizer()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        if model_type in [ModelType.LSTM]:
            print("number of parameters: %.2fM" % (self.n_params / 1e6,))

        self.save_at = self._determine_save_points(len(self.train_data))
        self.model_dir = self._setup_model_dir(sweep)
        self.save_csv = save_csv

    def _repeat_and_shuffle_data(self, data, num_repeats, shuffle_seed):
        random.seed(shuffle_seed)  # Initialize random seed for reproducibility
        
        unique_seeds = set()
        while len(unique_seeds) < num_repeats:
            unique_seeds.add(random.randint(1, 1000000))  # Generate unique seeds

        dfs = []
        for seed in unique_seeds:
            dfs.append(data.shuffle(data_shuffle_seed=seed).df)
        
        concatenated_df = pd.concat(dfs, ignore_index=True)
        return BracketsDataset(concatenated_df,
                            alphabet=data.tokenizer.alphabet, max_len=data.max_len, 
                            ylabels_heuristic=data.ylabels_heuristic,
                            dataset_info=data.dataset_info)
    
    def _initialize_model(self, model_type):
        if model_type == ModelType.Transformer:
            return get_transformer(
                n_layer=self.config["n_layer"],
                n_head=self.config["n_head"],
                n_embd=self.config["n_embd"],
                embd_pdrop=self.config["embd_pdrop"],
                resid_pdrop=self.config["resid_pdrop"],
                attn_pdrop=self.config["attn_pdrop"])
        elif model_type == ModelType.LSTM: 
            return get_LSTM(
                batch_size=self.batch_size, 
                num_layers=self.config["num_layers"],
                dropout_rate=self.config["dropout_rate"], 
                embedding_dim=self.config["embedding_dim"], 
                hidden_dim=self.config["hidden_dim"])
        else: 
            raise ValueError("Model type not supported or specified")

    def _get_optimizer(self):
        if self.config["opt"] == "sgd": 
            self.optimizer = SGD(self.model.parameters(), lr=self.config["lr"], 
                                 weight_decay=self.config["wd"])
        elif self.config["opt"] == "adam":
            self.optimizer = Adam(self.model.parameters(), lr=self.config["lr"], 
                                  weight_decay=self.config["wd"])
        else: 
            raise ValueError("Optimizer not supported or specified")
    
    def _set_seed(self, rdm_seed: int): 
        random.seed(rdm_seed)
        np.random.seed(rdm_seed)
        t.manual_seed(rdm_seed)
        t.cuda.manual_seed(rdm_seed)
        t.cuda.manual_seed_all(rdm_seed) # multi-GPU
        t.backends.cudnn.deterministic = True 
        t.backends.cudnn.benchmark = False
        t.use_deterministic_algorithms(True)

    def _seed_worker(self, worker_id):
        worker_seed = t.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    def _determine_save_points(self, train_size):
        save_at = sorted(set([i for i in [0, 100, 500, 1000] if i <= train_size] + 
                             [(i+1)*1000 for i in range(train_size//1000)] + [train_size]))
        return save_at

    def _setup_model_dir(self, sweep):
        model_dir = os.path.join(LOGS_DIR, "models", f"sweep_{self.run.sweep_id}" if sweep else "", 
                                 f"run_{self.run.id}")
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    def _print_initial_weights(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name}, Values: {param.data.cpu().numpy()}")
    
    def _get_preds(self, inputs: t.Tensor, input_strings: list[str]) -> t.Tensor:
        if isinstance(self.model, GPT): 
            outputs = self.model(inputs.cuda())[0]
            seq_lens = t.tensor([len(s) + 2 for s in input_strings])  
            last_indices = (seq_lens - 1).unsqueeze(1).repeat(1, N_CLASSES).cuda()
            preds = outputs.gather(1, last_indices.unsqueeze(1))[:, 0, :].cuda()
        elif isinstance(self.model, LSTMClassifier): 
            inputs = inputs.transpose(0, 1).cuda() 
            self.model.hidden = self.model.init_hidden(inputs.size(1))
            preds = self.model(inputs).cuda()
        
        return preds
    
    def train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        self.model.cuda().train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        
        for _, (input_strings, inputs, targets) in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            preds = self._get_preds(inputs, input_strings)
            loss = self.loss_fn(preds, targets.long().cuda())
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * targets.size(0)
            with t.no_grad():
                total_correct += ((preds[:, 1] > preds[:, 0]) == targets.bool().cuda()).sum().item()
            total_samples += targets.size(0)

        train_avg_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        return train_avg_loss, train_acc
    
    def train_model(self):
        self.model.cuda().train()
        
        summary_records = {"max_ood_acc": None, "min_ood_acc": None}
        for metric in ["_datapoints_seen", "_train_acc", "_indist_acc"]: 
            summary_records["max_ood" + metric] = None
            summary_records["min_ood" + metric] = None

        print("progress,datapoints_seen,train_loss,train_acc,indist_acc,ood_acc")
        for i in range(1, len(self.save_at)):
            records = {}

            train_dl = DataLoader(self.train_data[self.save_at[i-1]:self.save_at[i]], # type: ignore
                                  batch_size=self.batch_size, worker_init_fn=self._seed_worker, generator=self.g) # type: ignore
            
            train_loss, train_acc = self.train_epoch(train_dl)
            
            records.update({"datapoints_seen": self.save_at[i], "train_loss": train_loss, "train_acc": train_acc})

            indist_dl = DataLoader(self.indist_test_data, batch_size=16, worker_init_fn=self._seed_worker, generator=self.g)
            indist_test_loss, indist_test_acc = self.evaluate(indist_dl)
            records.update({"indist_test_loss": indist_test_loss, "indist_test_acc": indist_test_acc})
            
            ood_dl = DataLoader(self.ood_test_data, batch_size=16, worker_init_fn=self._seed_worker, generator=self.g)
            ood_test_loss, ood_test_acc = self.evaluate(ood_dl)
            records.update({"ood_test_loss": ood_test_loss, "ood_test_acc": ood_test_acc})

            wandb.log(records)
            print(f"[{i} of {len(self.save_at) - 1}],{self.save_at[i]},{train_loss:.3f},{train_acc:.3f},{indist_test_acc:.3f},{ood_test_acc:.3f}")

            if records["ood_test_acc"] > (summary_records["max_ood_acc"] if summary_records["max_ood_acc"] is not None else 0.0) and records["indist_test_acc"] > CUTOFF:
                summary_records.update({"max_ood_acc": records["ood_test_acc"], 
                                        "max_ood_datapoints_seen": records["datapoints_seen"],
                                        "max_ood_train_acc": records["train_acc"], 
                                        "max_ood_indist_acc": records["indist_test_acc"]})
                t.save(self.model.state_dict(), os.path.join(self.model_dir, "max_ood.pt"))
            
            if records["ood_test_acc"] < (summary_records["min_ood_acc"] if summary_records["min_ood_acc"] is not None else 1.0) and records["indist_test_acc"] > CUTOFF:
                summary_records.update({"min_ood_acc": records["ood_test_acc"], 
                                        "min_ood_datapoints_seen": records["datapoints_seen"],
                                        "min_ood_train_acc": records["train_acc"], 
                                        "min_ood_indist_acc": records["indist_test_acc"]})
                t.save(self.model.state_dict(), os.path.join(self.model_dir, "min_ood.pt"))

            if self.save_at[i] % self.len_train == 0:
                t.save(self.model.state_dict(), os.path.join(self.model_dir, f"run_{self.run.id}_checkpoint_{int(self.save_at[i] / self.len_train)}.pt"))
        
        wandb.save(os.path.join(self.model_dir, f"checkpoint_{int(self.save_at[i] / self.len_train)}.pt"))
        
        wandb.log(summary_records)
        if summary_records["max_ood_acc"] is not None and summary_records["min_ood_acc"] is not None:
            wandb.log({"diff_max_min_ood": summary_records["max_ood_acc"] - summary_records["min_ood_acc"]})
        else: 
            wandb.log({"diff_max_min_ood": None})
        
        wandb.log({"n_params": self.n_params, "save_csv": self.save_csv})
        
        self._save_summary(summary_records)
        if wandb.run is not None: 
            wandb.finish()
    
    def _save_summary(self, summary_records):
        to_save = pd.DataFrame([{"sweep_id": self.run.sweep_id, "run_id": self.run.id,
                                 **self.config, "max_ood_acc": summary_records["max_ood_acc"], 
                                 "min_ood_acc": summary_records["min_ood_acc"]}])
        df_path = os.path.join(LOGS_DIR, f"{self.save_csv}.csv")
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
        else:
            df = pd.DataFrame()
        df = pd.concat([df, to_save], ignore_index=True)
        df.to_csv(df_path, index=False)
    
    def evaluate(self, test_dl: DataLoader) -> tuple[float, float]:
        self.model.eval().cuda()
        
        net_loss, nCorrect, nTotal = 0.0, 0.0, 0.0
        with t.no_grad():
            for (input_strings, inputs, targets) in test_dl: 
                targets = targets.cuda().long()
                preds = self._get_preds(inputs, input_strings)
                
                loss = targets.size(0) * self.loss_fn(preds, targets)
                vec_equal = ((preds[:,1] > preds[:,0]) == targets.bool())
                
                net_loss += loss
                nCorrect += vec_equal.sum().item()
                nTotal += targets.size(0)

        loss = (net_loss / nTotal).item() # type: ignore
        acc = nCorrect / nTotal
        
        return loss, acc
    
LSTM_SWEEP_CFG = "configs/LSTM_sweep.yaml"
TRANSFORMER_SWEEP_CFG = "configs/transformer_sweep.yaml"

LSTM_CFG = "configs/LSTM.yaml"
TRANSFORMER_CFG = "configs/transformer.yaml"


TRAIN_PATH = "train_binomial(40,0.5).csv"
INDIST_PATH = "in_dist_test_binomial(40,0.5).csv"
OOD_PATH = "test_binomial(40,0.5).csv"
# SAVE_CSV = "1Mexp2_bin_40_05_Transformer" #! MAKE UNIQUE FOR EACH SET OF DATASETS
SAVE_CSV = "1Mexp2_bin_40_05_LSTM"

PROJECT_NAME = "hp_sweep"

def main(model_type: ModelType, project: str, entity: str, sweep: bool,
         sweep_id: str | None = None, data_repeat: int = 3):
    if not sweep and sweep_id is not None:
        raise ValueError("sweep_id should only be specified when sweep=True")

    if sweep_id is None and sweep:  # get new sweep config
        if model_type == ModelType.LSTM:
            config = yaml.safe_load(open(LSTM_SWEEP_CFG, "r"))
        elif model_type == ModelType.Transformer:
            config = yaml.safe_load(open(TRANSFORMER_SWEEP_CFG, "r"))

        sweep_id = wandb.sweep(config, project=project, entity=entity)

    train_data = BracketsDataset(pd.read_csv(TRAIN_PATH))
    indist_data = BracketsDataset(pd.read_csv(INDIST_PATH))
    ood_data = BracketsDataset(pd.read_csv(OOD_PATH))

    if sweep:
        train_fn = lambda: Trainer(model_type, num_repeats=data_repeat, save_csv=SAVE_CSV, sweep=sweep,
                                   train_data=train_data, indist_test_data=indist_data, ood_test_data=ood_data).train_model()
        # Start or resume wandb sweep
        wandb.agent(sweep_id, project=project, entity=entity, function=train_fn) # type: ignore
    else:
        Trainer(model_type=model_type, num_repeats=data_repeat, save_csv=SAVE_CSV, sweep=sweep,
                train_data=train_data, indist_test_data=indist_data, ood_test_data=ood_data).train_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model sweep with specified parameters.")
    parser.add_argument("model_type", type=str, choices=[e.value for e in ModelType], help="model type (Transformer or LSTM")
    parser.add_argument("--sweep", default=False, action="store_true", help="run a sweep")
    parser.add_argument("--sweep_id", type=str, default=None, help="sweep ID to resume an existing sweep")
    
    args = parser.parse_args()
    model_type = ModelType(args.model_type)

    main(model_type, sweep=args.sweep, project=PROJECT_NAME, entity="trdy", sweep_id=args.sweep_id, data_repeat=5)

