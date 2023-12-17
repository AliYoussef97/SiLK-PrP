from tqdm import tqdm
import torch
import numpy as np
from pathlib import Path
from pointnerf.utils.train_utils import move_to_device
from pointnerf.settings import CKPT_PATH
from pointnerf.losses.silk_loss import Loss
from pointnerf.model.backbone.model_utils import normalise_raw_descriptors
from pointnerf.utils.map_utils import get_correspondences
from pointnerf.utils.relative_pose import pose_estimation, relative_pose_error
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    """
    Training and validation.
    Inputs:
        config: (dict) configuration file
        model: (torch.nn.Module) model to train
        train_loader: (torch.utils.data.DataLoader) training data loader
        validation_loader: (torch.utils.data.DataLoader) validation data loader
        iteration: (int) iteration to start training from
        device: (str) device to use for training
    Outputs:
        None
    """
    def __init__(self, config, model, train_loader, validation_loader=None, iteration=0, device="cpu"):
        print(f'\033[92m🚀 Training started for {config["model"]["class_name"].upper()} model on {config["data"]["class_name"]}\033[0m')

        self.config = config
        
        self.model = model
        
        self.train_loader = train_loader
        
        self.validation_loader = validation_loader
                
        self.device = device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["training"]["learning_rate"])

        self.loss_fn = Loss(self.config["training"]["block_size"],
                            self.config["training"]["jax_device"],
                            self.config["training"]["temperature"],
                            self.config["training"]["feature_size"])
        
        self.checkpoint_name = self.config["ckpt_name"]
        self.checkpoint_path = Path(CKPT_PATH,self.checkpoint_name)
        self.checkpoint_path.mkdir(parents=True,exist_ok=True)

        self.writer = SummaryWriter(log_dir = Path(self.checkpoint_path,"logs"))

        self.iteration = iteration

        self.max_iterations = self.config["training"]["num_iters"]

        self.pbar = tqdm(desc="Training", total=self.max_iterations, colour="green")
        if self.iteration !=0: self.pbar.update(iter)
    
        self.train = True

        self.running_loss = []

        self.model.train()

        self.training()


    def training(self):
        while self.train:
            for batch in self.train_loader:
                self.train_step(batch)
                if self.iteration % self.config["save_or_validation_interval"] == 0:
                    self.save_val_step()
                if self.iteration == self.max_iterations:
                    self.save_checkpoint()
                    self.train = False
                    self.writer.flush()
                    self.writer.close()
                    self.pbar.close()
                    print(f'\033[92m✅ {self.config["model"]["model_name"].upper()} Training finished\033[0m')
                    break

  
    def train_step(self, batch):
        
        batch = move_to_device(batch, self.device)
        
        output = self.model(batch["raw"]["image"])

        warped_output = self.model(batch["warp"]["image"])

        logits_0, logits_1 = output["logits"], warped_output["logits"]

        desc_0, desc_1 = output["raw_descriptors"], warped_output["raw_descriptors"]

        desc_norm_0, desc_norm_1 = normalise_raw_descriptors(desc_0, self.config["model"]["scale_factor"]),\
                                   normalise_raw_descriptors(desc_1, self.config["model"]["scale_factor"])
        
        corr_0, corr_1 = get_correspondences(batch,
                                             self.config["training"]["feature_size"],
                                             bias=self.config["training"]["bias"],
                                             device=self.device)

        desc_loss, kpts_loss, precision, recall, mkpts_0, mkpts_1 = self.loss_fn(desc_norm_0,
                                                                                 desc_norm_1,
                                                                                 corr_0,
                                                                                 corr_1,
                                                                                 logits_0,
                                                                                 logits_1)
        P_est = pose_estimation(mkpts_0,
                                mkpts_1,
                                batch["camera_intrinsic_matrix"],
                                batch["camera_intrinsic_matrix"],
                                batch["gt_relative_pose"])

        rot_loss, transl_loss = relative_pose_error(P_est,
                                                    batch["gt_relative_pose"],
                                                    train=True)

        loss = desc_loss + kpts_loss + (rot_loss + transl_loss)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.running_loss.append(loss.item())

        self.writer.add_scalar("Descriptor loss", desc_loss, self.iteration)
        self.writer.add_scalar("Keypoints loss", kpts_loss, self.iteration)
        self.writer.add_scalar("Precision", precision, self.iteration)
        self.writer.add_scalar("Recall", recall, self.iteration)
        self.writer.add_scalar("Rotation loss", torch.rad2deg(rot_loss), self.iteration)
        self.writer.add_scalar("Translation loss", torch.rad2deg(transl_loss), self.iteration)
        self.writer.add_scalar("Total loss", loss, self.iteration)

        self.iteration += 1
        
        self.pbar.update(1)


    def save_val_step(self):
         
        self.running_loss = np.mean(self.running_loss)           
                
        if self.validation_loader is not None:
                 
            self.model.eval()
                    
            running_val_loss, val_precision, val_recall = self.validate()

            self.model.train()
                
            self.writer.add_scalar("Training Validation loss", running_val_loss, self.iteration)
            self.writer.add_scalar("Validation Precision", val_precision, self.iteration)
            self.writer.add_scalar("Validation Recall", val_recall, self.iteration)
            
            tqdm.write('Iteration: {}, Running Training loss: {:.4f}, Running Validation loss: {:.4f}, Validation Precision: {:.4f}, Validation Recall: {:.4f}'
                        .format(self.iteration, self.running_loss, running_val_loss, val_precision, val_recall))
        
        else:
            tqdm.write('Iteration: {}, Running Training loss: {:.4f}'
                       .format(self.iteration, self.running_loss))

        self.save_checkpoint()
        self.running_loss = []


    def save_checkpoint(self):
        torch.save({"iteration":self.iteration,
                    "model_state_dict":self.model.state_dict()},
                    Path(self.checkpoint_path,f'{self.checkpoint_name}_{self.iteration}.pth'))
    

    @torch.no_grad()         
    def validate(self):
        
        running_val_loss = []
        precision = []
        recall = []

        for val_batch in tqdm(self.validation_loader, desc="Validation",colour="blue"):

            val_batch = move_to_device(val_batch, self.device)
            
            val_output = self.model(val_batch["raw"]["image"])
            
            val_warped_output = self.model(val_batch["warp"]["image"])

            val_logits_0, val_logits_1 = val_output["logits"], val_warped_output["logits"]

            val_desc_0, val_desc_1 = val_output["raw_descriptors"], val_warped_output["raw_descriptors"]

            val_desc_norm_0, val_desc_norm_1 = normalise_raw_descriptors(val_desc_0,self.config["model"]["scale_factor"]),\
                                               normalise_raw_descriptors(val_desc_1,self.config["model"]["scale_factor"])  
            
            val_corr_0, val_corr_1 = get_correspondences(val_batch,
                                                         self.config["training"]["feature_size"],
                                                         bias=self.config["training"]["bias"])

            val_desc_loss, val_kpts_loss, val_precision, val_recall, val_mkpts_0, val_mkpts_1 = self.loss_fn(val_desc_norm_0,
                                                                                                             val_desc_norm_1,
                                                                                                             val_corr_0,
                                                                                                             val_corr_1,
                                                                                                             val_logits_0,
                                                                                                             val_logits_1)
            val_P_est = pose_estimation(val_mkpts_0,
                                        val_mkpts_1,
                                        val_batch["camera_intrinsic_matrix"],
                                        val_batch["camera_intrinsic_matrix"],
                                        val_batch["GT_relative_pose"])
            
            val_rot_loss, val_transl_loss = relative_pose_error(val_P_est,
                                                                val_batch["GT_relative_pose"],
                                                                train=False)
            
            val_loss = val_desc_loss + val_kpts_loss + (val_rot_loss + val_transl_loss)

            running_val_loss.append(val_loss.item())
            precision.append(val_precision)
            recall.append(val_recall)
        
        running_val_loss = np.mean(running_val_loss)
        precision = np.mean(precision)
        recall = np.mean(recall)
            
        return running_val_loss, precision, recall
