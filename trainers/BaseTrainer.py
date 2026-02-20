import torch

from .utils import calculate_EER, calculate_minDCF

class BaseTrainer:
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.device = device

    def save_model(self, path: str):
        """
        Save the model to the given path
        If model is a PyTorch model, it will be saved using torch.save(state_dict)
        Problem is when non-PyTorch model contains a Pytorch component (e.g. extractor). In that case,
        the trainer should implement custom saving/loading methods.

        param path: Path to save the model to
        """
        if isinstance(self.model, torch.nn.Module):
            torch.save(self.model.state_dict(), path)
        else:
            raise NotImplementedError(
                "Child classes for non-PyTorch models need to implement save_model method"
            )

    def load_model(self, path: str):
        """
        Load the model from the given path
        Try to load the model as a PyTorch model using torch.load,
        otherwise, the child class trainer should implement custom loading method.

        param path: Path to load the model from
        """
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)

            # Handle DDP 'module.' prefix mismatch
            is_ddp = isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                if is_ddp:
                    if not k.startswith("module."):
                        new_state_dict[f"module.{k}"] = v
                    else:
                        new_state_dict[k] = v
                else:
                    if k.startswith("module."):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v

            self.model.load_state_dict(new_state_dict)
        except FileNotFoundError:
            raise
        except Exception as e:  # Path correct, but not a PyTorch model
            # raise NotImplementedError(
            #     "Child classes for non-PyTorch models need to implement load_model method"
            # )
            raise e

    def calculate_EER(self, labels, predictions, plot_det: bool, det_subtitle: str) -> float:
        return calculate_EER(labels, predictions, name=type(self.model).__name__, plot_det=plot_det, det_subtitle=det_subtitle)
    
    def calculate_minDCF(self, labels, scores, p_target=0.95, c_miss=1, c_fa=10) -> float:
        return calculate_minDCF(labels, scores, p_target, c_miss, c_fa)

    def train(self, train_dataloader, val_dataloader, numepochs: int = 20):
        raise NotImplementedError("Child classes should implement the train method")
    
    def val(self, val_dataloader):
        raise NotImplementedError("Child classes should implement the val method")

    def eval(self, eval_dataloader, subtitle: str = ""):
        raise NotImplementedError("Child classes should implement the eval method")
