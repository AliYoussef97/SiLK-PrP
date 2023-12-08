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
    script = config["script"] # Name of model script. e.g. SuperPoint.py
    class_name = config["class_name"] # Name of class in model script. e.g. SuperPoint class

    model_script = importlib.import_module(f"pointnerf.model.{script}")
    model = getattr(model_script, class_name)(config)
        
    return model.to(device)