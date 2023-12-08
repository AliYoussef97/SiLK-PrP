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

def train_val(config: dict, 
              model: torch.nn.Module,
              train_loader: torch.utils.data.DataLoader,
              validation_loader: torch.utils.data.DataLoader = None, 
              iteration: int = 0,
              device: str = "cpu") -> None:
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

    print(f'\033[92m🚀 Training started for {config["model"]["model_name"].upper()} model on {config["data"]["class_name"]}\033[0m')

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    loss_fn = Loss(config["training"]["block_size"],
                   config["training"]["jax_device"],
                   config["training"]["temperature"],
                   config["training"]["feature_size"])

    checkpoint_name = config["ckpt_name"]
    checkpoint_path = Path(CKPT_PATH,checkpoint_name)
    checkpoint_path.mkdir(parents=True,exist_ok=True)   

    writer = SummaryWriter(log_dir = Path(checkpoint_path,"logs"))
 
    max_iterations = config["train"]["num_iters"]    
    iter = iteration

    pbar = tqdm(desc="Training", total=max_iterations, colour="green")
    if iter !=0: pbar.update(iter)
    
    running_loss = []
    
    Train = True

    model.train()

    while Train: 

        for batch in train_loader:

            batch = move_to_device(batch, device)
            
            output = model(batch["raw"]["image"])

            warped_output = model(batch["warp"]["image"])

            logits_0, logits_1 = output["logits"], warped_output["logits"]

            desc_0, desc_1 = output["raw_descriptors"], warped_output["raw_descriptors"]

            desc_norm_0, desc_norm_1 = normalise_raw_descriptors(desc_0, config["model"]["scale_factor"]),\
                                       normalise_raw_descriptors(desc_1, config["model"]["scale_factor"])
            
            corr_0, corr_1 = get_correspondences(batch,
                                                 config["training"]["feature_size"],
                                                 bias=config["training"]["bias"])

            desc_loss, kpts_loss, precision, recall, mkpts_0, mkpts_1 = loss_fn(desc_norm_0,
                                                                                desc_norm_1,
                                                                                corr_0,
                                                                                corr_1,
                                                                                logits_0,
                                                                                logits_1)
            P_est = pose_estimation(mkpts_0,
                                    mkpts_1,
                                    batch["camera_intrinsic_matrix"],
                                    batch["camera_intrinsic_matrix"],
                                    confidence)

            p_loss = relative_pose_error(P_est,
                                         batch["GT_relative_pose"],
                                         train=True)

            loss = desc_loss + kpts_loss + p_loss

            writer.add_scalar("Descriptor loss", desc_loss, iter)
            writer.add_scalar("Keypoints loss", kpts_loss, iter)
            writer.add_scalar("Precision", precision, iter)
            writer.add_scalar("Recall", recall, iter)
            writer.add_scalar("Pose loss", p_loss, iter)
            writer.add_scalar("Total loss", loss, iter)

            running_loss.append(loss.item())

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            iter += 1
            pbar.update(1)

            if iter % config["save_or_validation_interval"] == 0:
                
                running_loss = np.mean(running_loss)           
                
                if validation_loader is not None:
                    
                    model.eval()
                    
                    running_val_loss, precision, recall = validate(config, model, validation_loader, loss_fn, device)

                    model.train()
                            
                    writer.add_scalar("Tunning Validation loss", running_val_loss, iter)
                    writer.add_scalar("Validation Precision", precision, iter)
                    writer.add_scalar("Validation Recall", recall, iter)
                        
                    tqdm.write('Iteration: {}, Running Training loss: {:.4f}, Running Validation loss: {:.4f}, Precision: {:.4f}, Recall: {:.4f}'
                                .format(iter, running_loss, running_val_loss, precision, recall))
                
                else:
                   tqdm.write('Iteration: {}, Running Training loss: {:.4f}'
                              .format(iter, running_loss))

                torch.save({"iteration":iter,
                            "model_state_dict":model.state_dict()},
                            Path(checkpoint_path,f'{checkpoint_name}_{iter}.pth'))
                
                running_loss = []
                

            if iter == max_iterations:

                torch.save({"iteration":iter,
                            "model_state_dict":model.state_dict()},
                            Path(checkpoint_path,f'{checkpoint_name}_{iter}.pth'))
                Train = False
                writer.flush()
                writer.close()
                pbar.close()
                print(f'\033[92m✅ {config["model"]["model_name"].upper()} Training finished\033[0m')
                break

@torch.no_grad()         
def validate(config, model, validation_loader, loss_fn, device= "cpu"):
    
    running_val_loss = []
    precision = []
    recall = []

    for val_batch in tqdm(validation_loader, desc="Validation",colour="blue"):

        val_batch = move_to_device(val_batch, device)
        
        val_output = model(val_batch["raw"]["image"])
        
        val_warped_output = model(val_batch["warp"]["image"])

        val_logits_0, val_logits_1 = val_output["logits"], val_warped_output["logits"]

        val_desc_0, val_desc_1 = val_output["raw_descriptors"], val_warped_output["raw_descriptors"]

        val_desc_norm_0, val_desc_norm_1 = normalise_raw_descriptors(val_desc_0,config["model"]["scale_factor"]),\
                                           normalise_raw_descriptors(val_desc_1,config["model"]["scale_factor"])  
        
        val_corr_0, val_corr_1 = get_correspondences(val_batch,
                                                     config["training"]["feature_size"],
                                                     bias=config["training"]["bias"])

        val_desc_loss, val_kpts_loss, val_precision, val_recall, val_mkpts_0, val_mkpts_1 = loss_fn(val_desc_norm_0,
                                                                                                    val_desc_norm_1,
                                                                                                    val_corr_0,
                                                                                                    val_corr_1,
                                                                                                    val_logits_0,
                                                                                                    val_logits_1)
        val_P_est = pose_estimation(val_mkpts_0,
                                    val_mkpts_1,
                                    val_batch["camera_intrinsic_matrix"],
                                    val_batch["camera_intrinsic_matrix"],
                                    confidence)
        
        val_p_loss = relative_pose_error(val_P_est,
                                         val_batch["GT_relative_pose"],
                                         train=False)
        
        val_loss = val_desc_loss + val_kpts_loss + val_p_loss

        running_val_loss.append(val_loss.item())

        precision.append(val_precision)
        recall.append(val_recall)
    
    running_val_loss = np.mean(running_val_loss)
    precision = np.mean(precision)
    recall = np.mean(recall)
        
    return running_val_loss, precision, recall