"""
checkpointing, seems trivial for single gpu
case but maybe useful when we move to distributed?
might not matter, then we can move the logic into
the builders

note that torch.save can save a dict
look into torch.distributed.checkpoint
"""

import torch

from pathlib import Path

class Checkpointer:
    def __init__(self, checkpoint_save_path: str):
        self.checkpoint_save_path = checkpoint_save_path

        dir_path = Path(checkpoint_save_path)
        if dir_path.is_dir():
            print("checkpoint folder already exists")
        else:
            print("checkpoint folder does not exist, saving")
            dir_path.mkdir(parents=True, exist_ok=True)

    def save_model(self, model, step: int):
        model_save_path = self.checkpoint_save_path + f"/{step}/model.pt"
        torch.save(model.state_dict(), model_save_path)

    def save_dataloader(self, dataloader, step: int):
        dataloader_save_path = self.checkpoint_save_path + f"/{step}/dataloader.pt"
        torch.save(dataloader.state_dict(), dataloader_save_path)
        
    def save_optimizer(self, optimizer, step: int):
        optimizer_save_path = self.checkpoint_save_path + f"/{step}/optimizer.pt"
        torch.save(optimizer.state_dict(), optimizer_save_path)




