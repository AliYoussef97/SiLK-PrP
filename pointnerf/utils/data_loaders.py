import importlib
from torch.utils.data import DataLoader

def get_loader(config, task, device="cpu", validate_training=False):
    
    dataset = config["data"]["name"]
    class_name = config["data"]["class_name"]
    batch_size = config["data"]["batch_size"]

    data_script = importlib.import_module(f"silkprp.data.{dataset}")

    if task == "train":
        
        dataset = {"train":getattr(data_script,class_name)(config["data"], task="training", device=device)}

        data_loader = {"train": DataLoader(dataset["train"], 
                                           batch_size=batch_size,
                                           collate_fn=dataset["train"].batch_collator, 
                                           shuffle=True,
                                           pin_memory=True,
                                           persistent_workers=True,
                                           num_workers=4),
                       "validation":None}
        if validate_training: 

            dataset["validation"] = getattr(data_script, class_name)(config["data"], task="validation", device=device)

            data_loader["validation"] = DataLoader(dataset["validation"], 
                                                   batch_size=batch_size,
                                                   collate_fn=dataset["validation"].batch_collator,
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   persistent_workers=True,
                                                   num_workers=4)
    
    if task == "pose_evaluation" or task == "hpatches_evaluation":

        dataset = getattr(data_script, class_name)(config["data"], device=device)

        data_loader = DataLoader(dataset, 
                                 batch_size=batch_size,
                                 collate_fn=dataset.batch_collator,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=0)
    
    return data_loader