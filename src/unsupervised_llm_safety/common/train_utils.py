import logging
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader
from tqdm import tqdm

log = logging.getLogger(__name__)


def train_model(
    cfg: DictConfig,
    model: nn.Module,
    train_dataloader: DataLoader,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Generic training function for models.

    The model computes its own loss in forward() when labels are
    provided, so no external loss function is needed.

    Args:
        cfg: Configuration dictionary
        model: Model to train (must implement forward(x, label=y))
        train_dataloader: DataLoader for training data
        device: Device to train on (defaults to CUDA if available)

    Returns:
        Trained model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Validate learning rate configuration
    if not hasattr(cfg, "training") or not hasattr(
        cfg.training, "learning_rate"
    ):
        raise ValueError(
            "Learning rate must be specified at cfg.training.learning_rate"
        )
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # Model-specific training start hook (e.g., activation monitoring setup)
    if hasattr(model, "on_training_start"):
        model.on_training_start(
            cfg=cfg,
            train_dataloader=train_dataloader,
            device=device,
            num_epochs=cfg.num_epochs,
        )

    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0

        pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{cfg.num_epochs}",
            leave=True,
            ncols=100,
        )

        for batch in pbar:
            inputs, labels, _ = batch

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            optimizer.zero_grad()

            # Models compute their own loss in forward()
            outputs = model(inputs, label=labels, reduction="mean")
            loss = outputs.loss

            if batch_count == 0 and epoch % 10 == 0:
                log.info(f"Batch loss: {loss}")

            loss.backward()

            # Log detailed metrics if available
            if outputs.metrics and batch_count == 0 and epoch % 1 == 0:
                if "reconstruction_loss" in outputs.metrics:
                    log.info(
                        f"Epoch {epoch+1}, Reconstruction loss: "
                        f"{outputs.metrics['reconstruction_loss']:.6f}"
                    )
                if "bce_loss" in outputs.metrics:
                    log.info(
                        f"Epoch {epoch+1}, BCE loss: "
                        f"{outputs.metrics['bce_loss']:.6f}"
                    )
                if "contrastive_loss" in outputs.metrics:
                    log.info(
                        f"Epoch {epoch+1}, Contrastive loss: "
                        f"{outputs.metrics['contrastive_loss']:.6f}"
                    )
                if "median_l0" in outputs.metrics:
                    log.info(
                        f"Epoch {epoch+1}, L0: "
                        f"median={outputs.metrics['median_l0']:.1f}, "
                        f"mean={outputs.metrics['mean_l0']:.1f}"
                    )

            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

            avg_loss = total_loss / batch_count
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        avg_loss = total_loss / max(1, len(train_dataloader))
        log.info(f"Epoch {epoch+1}/{cfg.num_epochs}, Avg Loss: {avg_loss:.6f}")

        # Model-specific epoch-end hook (e.g., activation monitoring)
        if hasattr(model, "on_epoch_end"):
            model.on_epoch_end(
                epoch=epoch,
                num_epochs=cfg.num_epochs,
                train_dataloader=train_dataloader,
                device=device,
                avg_loss=avg_loss,
                total_loss=total_loss,
                batch_count=batch_count,
            )

    return model


def create_model(cfg: DictConfig, hs_data: dict) -> nn.Module:
    """Create model based on config using Hydra instantiation.

    Args:
        cfg: Configuration dictionary containing 'model' key with model config
        hs_data: Hidden states data dictionary containing train/test data.
                Expected keys: 'train' or 's1' with 'hidden_states' tensor

    Returns:
        Initialized model ready for training
    """
    import hydra
    from omegaconf import OmegaConf

    if not hasattr(cfg, "model"):
        raise ValueError("Configuration must contain 'model' key")

    if not hasattr(cfg.model, "_target_"):
        raise AttributeError(
            "Model config must have '_target_' field pointing to model "
            "class. Example: _target_: "
            "unsupervised_llm_safety.models.autoencoder.Autoencoder"
        )

    # Get training data for dimension inference
    if "train" in hs_data:
        train_data = hs_data["train"]
    elif "s1" in hs_data:
        train_data = hs_data["s1"]
    else:
        raise ValueError(
            "No suitable training data found in hs_data. "
            "Expected 'train' or 's1' key."
        )

    if "hidden_states" not in train_data:
        raise ValueError("Training data must contain 'hidden_states' tensor")

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    input_dim = train_data["hidden_states"].shape[-1]

    if "input_dim" in model_cfg:
        model_cfg["input_dim"] = input_dim

    # Remove non-model parameters that might be in the config
    for param in [
        "learning_rate",
        "lr",
        "optimizer",
        "scheduler",
        "weight_decay",
        "momentum",
        "betas",
        "eps",
    ]:
        model_cfg.pop(param, None)

    log.info(f"Creating model: {model_cfg.get('_target_', 'Unknown')}")

    try:
        model = hydra.utils.instantiate(model_cfg)
    except Exception as e:
        log.error(f"Failed to instantiate model: {e}")
        log.error(f"Model config: {model_cfg}")
        raise

    log.info(
        f"Successfully created {model.__class__.__name__} model with "
        f"{sum(p.numel() for p in model.parameters())} parameters"
    )

    return model


def load_model_and_tokenizer(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = (
        tokenizer.unk_token
        if tokenizer.pad_token is None
        else tokenizer.pad_token
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    check_random_state(seed)
