"""Test model implementations."""

import pytest
import torch

# Import models to register them
from src.models.transformers import informer, autoformer
from src.models.neural_ode import latent_ode, ode_rnn
from src.models.registry import get_model


def get_test_config():
    """Get minimal test configuration."""
    return {
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        "seq_len": 96,
        "label_len": 48,
        "pred_len": 24,
        "model": {
            "d_model": 64,
            "n_heads": 4,
            "e_layers": 1,
            "d_layers": 1,
            "d_ff": 128,
            "dropout": 0.1,
            "factor": 3,
        },
        "data": {
            "freq": "h",
        },
    }


def test_informer_forward():
    """Test Informer forward pass."""
    config = get_test_config()
    config["model"]["name"] = "informer"
    config["model"]["attn"] = "prob"

    model = get_model("informer", config)
    model.eval()

    batch_size = 2
    x_enc = torch.randn(batch_size, config["seq_len"], config["enc_in"])
    x_mark_enc = torch.randn(batch_size, config["seq_len"], 4)
    x_dec = torch.randn(batch_size, config["label_len"] + config["pred_len"], config["dec_in"])
    x_mark_dec = torch.randn(batch_size, config["label_len"] + config["pred_len"], 4)

    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    assert output.shape == (batch_size, config["pred_len"], config["c_out"])
    assert not torch.isnan(output).any()


def test_autoformer_forward():
    """Test Autoformer forward pass."""
    config = get_test_config()
    config["model"]["name"] = "autoformer"
    config["model"]["moving_avg"] = 25

    model = get_model("autoformer", config)
    model.eval()

    batch_size = 2
    x_enc = torch.randn(batch_size, config["seq_len"], config["enc_in"])
    x_mark_enc = torch.randn(batch_size, config["seq_len"], 4)
    x_dec = torch.randn(batch_size, config["label_len"] + config["pred_len"], config["dec_in"])
    x_mark_dec = torch.randn(batch_size, config["label_len"] + config["pred_len"], 4)

    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    assert output.shape == (batch_size, config["pred_len"], config["c_out"])
    assert not torch.isnan(output).any()


def test_latent_ode_forward():
    """Test Latent ODE forward pass."""
    config = get_test_config()
    config["model"] = {
        "name": "latent_ode",
        "latent_dim": 32,
        "rec_dims": 32,
        "ode_dims": 64,
        "dec_dims": 32,
        "rec_layers": 1,
        "ode_layers": 1,
        "dec_layers": 1,
    }
    config["ode"] = {
        "solver": "dopri5",
        "rtol": 1e-3,
        "atol": 1e-4,
        "use_adjoint": False,  # Faster for testing
    }

    model = get_model("latent_ode", config)
    model.eval()

    batch_size = 2
    x_enc = torch.randn(batch_size, config["seq_len"], config["enc_in"])
    x_mark_enc = torch.randn(batch_size, config["seq_len"], 4)
    x_dec = torch.randn(batch_size, config["label_len"] + config["pred_len"], config["dec_in"])
    x_mark_dec = torch.randn(batch_size, config["label_len"] + config["pred_len"], 4)

    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    assert output.shape == (batch_size, config["pred_len"], config["c_out"])
    assert not torch.isnan(output).any()


def test_model_parameter_count():
    """Test parameter counting."""
    config = get_test_config()
    config["model"]["name"] = "informer"

    model = get_model("informer", config)
    param_count = model.get_parameter_count()

    assert param_count > 0
    assert isinstance(param_count, int)


def test_model_type_property():
    """Test model type property."""
    config = get_test_config()

    # Standard model
    config["model"]["name"] = "informer"
    model = get_model("informer", config)
    assert model.model_type == "standard"

    # ODE model
    config["model"] = {"name": "latent_ode", "latent_dim": 32}
    config["ode"] = {"use_adjoint": False}
    model = get_model("latent_ode", config)
    assert model.model_type == "ode"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
