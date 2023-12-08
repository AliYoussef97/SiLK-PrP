import tyro
import yaml
import torch
from pathlib import Path
from typing import Literal
from dataclasses import dataclass 
from pointnerf.settings import CKPT_PATH
from pointnerf.utils.get_model import get_model
from pointnerf.utils.data_loaders import get_loader
from pointnerf.engine_solvers.train import train_val


@dataclass
class options:
    """
    Training options.
    Args:
        validate_training: Validate during training.
    """
    validate_training: bool = False


@tyro.conf.configure(tyro.conf.FlagConversionOff)
class main():
    """main class, script backbone.
    Args:
        config_path: Path to configuration.
        task: The task to be performed.
    """
    def __init__(self,
                 config_path: str,
                 task: Literal["train"],
                 training:options) -> None:

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f) 
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if task == "train":

            self.model = get_model(self.config["model"], device=self.device)
            
            self.dataloader = get_loader(self.config, 
                                         task, 
                                         device="cpu", 
                                         validate_training=training.validate_training)

            if self.config["pretrained"]:
                
                model_state_dict =  self.model.state_dict()
                
                pretrained_dict = torch.load(Path(CKPT_PATH,self.config["pretrained"]), map_location=self.device)
                pretrained_state = pretrained_dict["model_state_dict"]
                
                for k,v in pretrained_state.items():
                    if k in model_state_dict.keys():
                        model_state_dict[k] = v
                
                self.model.load_state_dict(model_state_dict)
                print(f'\033[92m✅ Loaded pretrained model \033[0m')

                self.iteration = pretrained_dict["iteration"]

            self.train()
    
    def train(self):

        if self.config["continue_training"]:
            iteration = self.iteration
        else:
            iteration = 0
        
        train_val(self.config, self.model, 
                  self.dataloader["train"], self.dataloader["validation"], 
                  iteration, self.device)
        
if __name__ == '__main__':
    tyro.cli(main)