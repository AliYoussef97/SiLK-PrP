
import tyro
import yaml
import torch
from pathlib import Path
from typing import Literal
from dataclasses import dataclass 
from pointnerf.settings import CKPT_PATH
from pointnerf.utils.get_model import get_model
from pointnerf.utils.data_loaders import get_loader
from pointnerf.engine_solvers.train import Trainer


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

        self.config = yaml.load(stream=open(config_path, 'r'), Loader=yaml.FullLoader)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

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
            
            if self.config["continue_training"]:
                iteration = self.iteration
            else:
                iteration = 0

        Trainer(config=self.config,
                model=self.model,
                train_loader=self.dataloader["train"],
                validation_loader=self.dataloader["validation"],
                iteration=iteration,
                device=self.device)
        


if __name__ == '__main__':
    tyro.cli(main)