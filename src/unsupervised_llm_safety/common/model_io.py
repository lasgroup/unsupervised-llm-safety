"""Simple, unified model I/O functions."""

import logging
from typing import Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_model(model, filepath):
    """Save entire model object, not state dict."""
    torch.save(model, filepath)


def load_model(filepath, device="cuda"):
    """Load entire model object (full model, not just encoder).

    This function loads the complete model as saved. For models with
    encoder/decoder structure (e.g., Autoencoder), it returns the
    full model object.

    If you only need the encoder, use load_encoder() instead.
    """
    model = torch.load(filepath, map_location=device, weights_only=False)
    return model.to(device)


def infer_model_key_from_path(model_path: str) -> Optional[str]:
    """Try to infer model key from model path.

    Args:
        model_path: Path to the model file

    Returns:
        Model key if found, None otherwise
    """
    # Common patterns in paths
    path_patterns = {
        "qwen": "qwen2_1.5b",
        "ministral": "ministral_8b",
        "llama": "llama3_8b",
    }

    path_lower = model_path.lower()
    for pattern, model_key in path_patterns.items():
        if pattern in path_lower:
            logger.info(f"Inferred model key '{model_key}' from path")
            return model_key

    logger.warning(f"Could not infer model key from path: {model_path}")
    return None


def load_encoder(
    encoder_path: str,
    device: Union[str, torch.device] = "cuda",
    pretrained_encoder_path: Optional[str] = None,
    apply_cvdd_scaler: bool = True,
) -> nn.Module:
    """Load encoder from checkpoint file.

    Tries to load directly first (simple torch.load), only reconstructs if necessary.

    Args:
        encoder_path: Path to the encoder checkpoint file
        device: Device to load the encoder to (default: "cuda")
        pretrained_encoder_path: Optional path to original pretrained encoder
            (used for CVDD checkpoints that used pretrained encoders)
        apply_cvdd_scaler: Whether to apply CVDD's StandardScaler wrapper
            (only relevant for CVDD checkpoints, default: True)

    Returns:
        Loaded encoder module (nn.Module)

    Raises:
        ValueError: If checkpoint format is not recognized
    """
    if isinstance(device, str):
        device = torch.device(device)

    logger.info(f"Loading encoder from {encoder_path}")

    # Load checkpoint
    checkpoint = torch.load(
        encoder_path, map_location=device, weights_only=False
    )

    # Case 1: Direct model object with encoder attribute (most common - Autoencoder)
    if hasattr(checkpoint, "encoder") and hasattr(checkpoint, "forward"):
        logger.info(
            "Checkpoint contains a model object with encoder attribute"
        )
        encoder = checkpoint.encoder.to(device)
        encoder.eval()
        return encoder

    # Case 2: Dictionary with encoder key (saved encoder directly)
    if isinstance(checkpoint, dict) and "encoder" in checkpoint:
        logger.info("Checkpoint contains 'encoder' key")
        encoder = checkpoint["encoder"]
        if isinstance(encoder, nn.Module):
            encoder = encoder.to(device)
            encoder.eval()
            return encoder

    # Case 3: Direct nn.Module (encoder saved directly)
    if isinstance(checkpoint, nn.Module):
        logger.info("Checkpoint is a direct nn.Module")
        encoder = checkpoint.to(device)
        encoder.eval()
        return encoder

    # Case 4: CVDD or CVDD L0 checkpoint - try to load from pretrained_encoder_path first
    if isinstance(checkpoint, dict) and checkpoint.get("method_name") in [
        "cvdd",
        "cvdd_l0",
    ]:
        logger.info(f"Loading {checkpoint.get('method_name')} checkpoint")

        # If pretrained_encoder_path is provided, just load it directly (simple!)
        if pretrained_encoder_path:
            logger.info(
                f"Loading pretrained encoder from {pretrained_encoder_path}"
            )
            pretrained = torch.load(
                pretrained_encoder_path,
                map_location=device,
                weights_only=False,
            )
            if hasattr(pretrained, "encoder"):
                encoder = pretrained.encoder.to(device)
            elif isinstance(pretrained, dict) and "encoder" in pretrained:
                encoder = pretrained["encoder"].to(device)
            elif isinstance(pretrained, nn.Module):
                encoder = pretrained.to(device)
            else:
                raise ValueError(
                    "Could not extract encoder from pretrained_encoder_path. "
                    "Expected model with .encoder attribute, dict with 'encoder' key, or nn.Module"
                )
        else:
            # Fall back to reconstruction only if no pretrained path provided
            logger.warning(
                "No pretrained_encoder_path provided for CVDD checkpoint. "
                "Attempting reconstruction from state dict (may be unreliable)."
            )
            encoder = _load_cvdd_encoder_simple(
                checkpoint, device, apply_cvdd_scaler
            )

        encoder.eval()
        return encoder

    # Case 5: Dictionary with model_state_dict - try simple approach first
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Check if there's a saved encoder somewhere we can use
        if pretrained_encoder_path:
            logger.info(
                f"Checkpoint has model_state_dict, but using pretrained_encoder_path instead: {pretrained_encoder_path}"
            )
            pretrained = torch.load(
                pretrained_encoder_path,
                map_location=device,
                weights_only=False,
            )
            if hasattr(pretrained, "encoder"):
                encoder = pretrained.encoder.to(device)
                encoder.eval()
                return encoder

        # Otherwise, warn and suggest using pretrained_encoder_path
        raise ValueError(
            "Checkpoint contains 'model_state_dict' but no direct encoder found. "
            "Please provide pretrained_encoder_path to load the encoder directly, "
            "or save the encoder separately using: torch.save(model.encoder, 'encoder.pt')"
        )

    # Unknown format
    raise ValueError(
        f"Unknown checkpoint format. "
        f"Type: {type(checkpoint)}, "
        f"Keys (if dict): {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'N/A'}. "
        "Please save the encoder directly using: torch.save(model.encoder, 'encoder.pt')"
    )


def _load_cvdd_encoder_simple(
    checkpoint: dict,
    device: torch.device,
    apply_cvdd_scaler: bool,
) -> nn.Module:
    """Simple CVDD encoder loading - only for DeepSVDD (trained from scratch).

    For CVDD with pretrained encoders, use pretrained_encoder_path instead.
    """
    state = checkpoint["state"]
    config = checkpoint["config"]
    net_state_keys = list(state["net_state_dict"].keys())

    # Only handle DeepSVDD (trained from scratch) - simple case
    if "fc1.weight" in net_state_keys:
        logger.info("CVDD was trained from scratch (DeepSVDDNetwork)")
        encoder = _load_deepsvdd_encoder(state, config, device)
    else:
        raise ValueError(
            f"CVDD checkpoint uses pretrained encoder but no pretrained_encoder_path provided. "
            f"Please provide pretrained_encoder_path to load the encoder directly. "
            f"State dict keys: {net_state_keys[:5]}..."
        )

    # Optionally wrap with CVDD's scaler
    if apply_cvdd_scaler and "scaler_mean" in state:
        logger.info("Wrapping encoder with CVDD's StandardScaler")
        encoder = _wrap_encoder_with_scaler(encoder, state, device)

    return encoder


def _load_deepsvdd_encoder(
    state: dict, config: dict, device: torch.device
) -> nn.Module:
    """Load DeepSVDDNetwork encoder from CVDD state dict."""
    from unsupervised_llm_safety.anodet_algos.deep_svdd import DeepSVDDNetwork

    net = DeepSVDDNetwork(
        input_dim=state["input_dim"], rep_dim=config["rep_dim"]
    ).to(device)

    net.load_state_dict(state["net_state_dict"])
    logger.info(
        f"Reconstructed DeepSVDDNetwork: {state['input_dim']} -> {config['rep_dim']}"
    )
    return net


def _load_sequential_encoder_from_state_dict(
    net_state_dict: dict, device: torch.device
) -> nn.Module:
    """Reconstruct Sequential/Autoencoder encoder from state_dict.

    Args:
        net_state_dict: State dict with keys like "model.0.weight", "model.0.bias", etc.
        device: Device to load encoder to

    Returns:
        Reconstructed Sequential encoder module
    """
    net_state_keys = list(net_state_dict.keys())
    logger.info(
        f"Reconstructing Sequential encoder from state_dict (keys: {net_state_keys[:5]}...)"
    )

    # Build Sequential model from state_dict
    # The original encoder (e.g. SAE) typically has ReLU activations after
    # each linear layer. These aren't stored in the state_dict since they
    # have no parameters, so we must reconstruct them.
    linear_layers = []
    i = 0
    while f"model.{i}.weight" in net_state_dict:
        weight = net_state_dict[f"model.{i}.weight"]
        bias = net_state_dict.get(f"model.{i}.bias")
        in_features = weight.shape[1]
        out_features = weight.shape[0]
        linear = nn.Linear(in_features, out_features, bias=bias is not None)
        linear_layers.append(linear)
        i += 1

    if not linear_layers:
        raise ValueError(
            f"Could not reconstruct Sequential encoder from state_dict. "
            f"Keys: {net_state_keys[:10]}"
        )

    # Reconstruct with ReLU after each linear layer (matching SAE encoder
    # architecture). The state_dict only stores parameterized layers, so
    # activation functions must be re-added during reconstruction.
    layers = []
    for linear in linear_layers:
        layers.append(linear)
        layers.append(nn.ReLU())

    encoder = nn.Sequential(*layers).to(device)

    # Strip "model." prefix from keys to match Sequential's expected format
    sequential_state_dict = {}
    for key, value in net_state_dict.items():
        if key.startswith("model."):
            new_key = key.replace("model.", "")
            sequential_state_dict[new_key] = value
        else:
            sequential_state_dict[key] = value

    encoder.load_state_dict(sequential_state_dict)
    logger.info(f"Reconstructed Sequential encoder with {len(layers)} layers")
    return encoder


def reconstruct_encoder_from_state_dict(
    net_state_dict: dict,
    input_dim: int,
    rep_dim: int,
    device: Union[str, torch.device],
) -> nn.Module:
    """Reconstruct encoder from state_dict (handles both DeepSVDD and Sequential/Autoencoder).

    Args:
        net_state_dict: State dict of the encoder network
        input_dim: Input dimension
        rep_dim: Representation dimension
        device: Device to load encoder to

    Returns:
        Reconstructed encoder module
    """
    if isinstance(device, str):
        device = torch.device(device)

    net_state_keys = list(net_state_dict.keys())

    # Check if it's DeepSVDDNetwork (trained from scratch)
    if "fc1.weight" in net_state_keys:
        # Use existing helper for DeepSVDD
        state = {"net_state_dict": net_state_dict, "input_dim": input_dim}
        config = {"rep_dim": rep_dim}
        return _load_deepsvdd_encoder(state, config, device)
    else:
        # Sequential/Autoencoder
        return _load_sequential_encoder_from_state_dict(net_state_dict, device)


def _wrap_encoder_with_scaler(
    encoder: nn.Module, state: dict, device: torch.device
) -> nn.Module:
    """Wrap encoder with StandardScaler to match CVDD training conditions."""
    scaler_mean = torch.from_numpy(state["scaler_mean"]).float().to(device)
    scaler_scale = torch.from_numpy(state["scaler_scale"]).float().to(device)

    class ScaledEncoder(nn.Module):
        """Wrapper that applies StandardScaler before encoder."""

        def __init__(
            self, encoder: nn.Module, mean: torch.Tensor, scale: torch.Tensor
        ):
            super().__init__()
            self.encoder = encoder
            self.register_buffer("mean", mean)
            self.register_buffer("scale", scale)

        def forward(self, x):
            # Apply scaling: (x - mean) / scale
            x_scaled = (x - self.mean) / self.scale
            return self.encoder(x_scaled)

    return ScaledEncoder(encoder, scaler_mean, scaler_scale)
