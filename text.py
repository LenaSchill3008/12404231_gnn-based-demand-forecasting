import torch

ckpt = torch.load("/Users/lenaschill/Desktop/12404231_gnn-based-demand-forecasting/model_checkpoints/model_checkpoint_e50.pt", map_location="cpu")
print("Epoch:", ckpt["epoch"])
print("Loss:", ckpt["loss"])