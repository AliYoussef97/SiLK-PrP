import importlib
from torch.utils.data import DataLoader

def get_loader(config, task, device="cpu", validate_training=False):
    
    dataset = config["data"]["name"] # Name of dataset script. e.g. Synthetic_data.py
    class_name = config["data"]["class_name"] # Name of class in dataset script. e.g. SyntheticShapes class
    batch_size = config["data"]["batch_size"]

    data_script = importlib.import_module(f"pointnerf.data.{dataset}")

    if task == "train":
        
        dataset = {"train":getattr(data_script,class_name)(config["data"], task="training", device=device)}

        data_loader = {"train": DataLoader(dataset["train"], 
                                        batch_size=batch_size,
                                        collate_fn=dataset["train"].batch_collator, 
                                        shuffle=True,
                                        pin_memory=True,
                                        persistent_workers=True,
                                        num_workers=2),
                        "validation":None}
        if validate_training: 

            dataset["validation"] = getattr(data_script, class_name)(config["data"], task="validation", device=device)

            data_loader["validation"] = DataLoader(dataset["validation"], 
                                                batch_size=batch_size,
                                                collate_fn=dataset["validation"].batch_collator,
                                                shuffle=False,
                                                pin_memory=True,
                                                persistent_workers=True,
                                                num_workers=2)
    
    return data_loader