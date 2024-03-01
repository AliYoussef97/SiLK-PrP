
import tyro
import yaml
import torch
from pathlib import Path
from typing import Literal
from dataclasses import dataclass 
from silkprp.settings import CKPT_PATH
from silkprp.utils.get_model import get_model, load_checkpoint
from silkprp.utils.data_loaders import get_loader
from silkprp.engine_solvers.train import Trainer
from silkprp.evaluations.pose_evaluation import estimate_pose_errors
from evaluations.hpatches_evaluation import estimate_hpatches_metrics


@dataclass
class options:
    """
    Training options.
    Args:
        validate_training: Validate during training.
    """
    validate_training: bool = False

@dataclass
class pose_options:
    """Pose evaluation options.

    Args:
        validate_training: configuation path
    """
    shuffle: bool = False
    max_length: int = -1


@dataclass
class hpatches_options:
    """HPatches evaluation options.

    Args:
        validate_training: configuation path
    """
    alteration: Literal["i", "v", "all"] = "v"
    

@tyro.conf.configure(tyro.conf.FlagConversionOff)
class main():
    """main class, script backbone.
    Args:
        config_path: Path to configuration.
        task: The task to be performed.
    """
    def __init__(self,
                 config_path: str,
                 task: Literal["train", "pose_evaluation", "hpatches_evaluation"],
                 training:options,
                 pose:pose_options,
                 hpatches:hpatches_options) -> None:

        self.config = yaml.load(stream=open(config_path, 'r'), Loader=yaml.FullLoader)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if task == "train":

            self.model = get_model(self.config["model"], device=self.device)
            
            self.dataloader = get_loader(self.config, 
                                         task, 
                                         device = "cpu", 
                                         validate_training = training.validate_training)

            if self.config["pretrained"]:
                                
                pretrained_dict = torch.load(Path(CKPT_PATH, self.config["pretrained"]), map_location=self.device)

                self.model = load_checkpoint(self.model, pretrained_dict, eval=False)

            
            if self.config["continue_training"]:
                self.iteration = pretrained_dict["iteration"]
                self.opt = pretrained_dict["optimizer_state_dict"]
            else:
                self.iteration = 0
                self.opt = None

            Trainer(config=self.config,
                    model=self.model,
                    train_loader=self.dataloader["train"],
                    validation_loader=self.dataloader["validation"],
                    iteration=self.iteration,
                    optimizer_state_dict=self.opt,
                    device=self.device)
            
        if task == "pose_evaluation" or task == "hpatches_evaluation":

            self.model = get_model(self.config["model"], device=self.device)

            pretrained_dict = torch.load(Path(CKPT_PATH, self.config["pretrained"]), map_location=self.device)

            self.model = load_checkpoint(self.model, pretrained_dict, eval=True)

            if task == "pose_evaluation":
                if pose.shuffle:
                    self.config["data"]["shuffle"] = True
            
                if pose.max_length > -1:
                    self.config["data"]["max_length"] = pose.max_length
                
                self.dataloader = get_loader(self.config,
                                             task,
                                             device = self.device,
                                             validate_training = False)
            
                estimate_pose_errors(self.config, 
                                     self.model, 
                                     self.dataloader, 
                                     self.device)
                
            if task == "hpatches_evaluation":
                
                self.config["data"]["alteration"] = hpatches.alteration
                
                self.dataloader = get_loader(self.config,
                                             task,
                                             device = self.device,
                                             validate_training = False)
            
                estimate_hpatches_metrics(self.config, 
                                          self.model, 
                                          self.dataloader, 
                                          self.device)

if __name__ == '__main__':
    tyro.cli(main)