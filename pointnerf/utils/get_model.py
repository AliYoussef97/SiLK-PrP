import importlib

def get_model(config: dict, 
              device: str = "cpu"):
    """
    Get Model from config.
    Input:
        config: dict, configuration for model
        device: str, device to run model on
    Output:
        model: nn.Module, model
    """
    script = config["script"] # Name of script in model folder. e.g. PointNeRF.py
    class_name = config["class_name"] # Name of class in model script. e.g. Pointnerf

    model_script = importlib.import_module(f"pointnerf.model.{script}")
    model = getattr(model_script, class_name)(config)
        
    return model.to(device)