import torch
from model_def import Net_v4

def check_model(model_path):
    print(f"Checking model at: {model_path}")
    
    try:
        # Try loading the model
        loaded = torch.load(model_path, map_location='cpu')
        print(f"Loaded object type: {type(loaded)}")
        
        if isinstance(loaded, dict):
            print("Model was saved as state dict")
            print("Keys:", loaded.keys())
        elif isinstance(loaded, Net_v4):
            print("Model was saved as full model")
            print("State dict keys:", loaded.state_dict().keys())
        else:
            print("Unknown model format")
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")

if __name__ == "__main__":
    check_model("model/net_bicycle_model_100ms_20000_v4.model")