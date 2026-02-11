import wandb
import numpy as np

def log_vector_snapshot(tag, vec, step):
    """
    Logs a (dim, value) table so you can drop a bar/line chart in the UI.
    """
    dims  = np.arange(len(vec))
    table = wandb.Table(data=np.stack([dims, vec], axis=1),
                        columns=["dim", "value"])
    wandb.log({f"{tag}_snapshot": table}, step=step)


def log_vector_heatmap(tag, vec, step):
    wandb.log({tag: wandb.Image(
        vec[None, :],           # shape (1, 768)
        caption=f"{tag} @ step {step}"
    )})
