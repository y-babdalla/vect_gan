"""This file contains unit tests for the VectGan class and its components."""

import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vect_gan.synthesizers.vectgan import (
    VAE,
    Decoder,
    Discriminator,
    Encoder,
    Generator,
    Residual,
    VectGan,
)
from vect_gan.utils.data_sampler import DataSampler
from vect_gan.utils.data_transformer import DataTransformer


@pytest.fixture
def mock_transformer() -> MagicMock:
    """Fixture that returns a mocked DataTransformer instance with properties needed by VectGan."""
    transformer = MagicMock(spec=DataTransformer)
    transformer.output_dimensions = 10
    transformer.output_info_list = [[{"dim": 10, "activation_fn": "tanh"}]]
    return transformer


@pytest.fixture
def mock_data_sampler() -> MagicMock:
    """Fixture that returns a mocked DataSampler instance with properties needed by VectGan."""
    data_sampler = MagicMock(spec=DataSampler)
    data_sampler.dim_cond_vec.return_value = 0
    data_sampler.sample_condvec.return_value = None
    data_sampler.sample_original_condvec.return_value = None
    return data_sampler


@pytest.fixture
def simple_data() -> np.ndarray:
    """Fixture that creates a simple random dataset for testing training."""
    return np.random.randn(100, 10).astype(np.float32)


def test_residual_layer() -> None:
    """Test that the Residual layer can be instantiated and run a forward pass."""
    layer = Residual(i=8, o=16)
    x = torch.randn(5, 8)  # Batch size 5, input dim 8
    out = layer(x)
    assert out.shape == (5, 8 + 16), "Residual output shape incorrect."


def test_encoder() -> None:
    """Test that the Encoder can be instantiated and run a forward pass."""
    encoder = Encoder(input_dim=10, encoder_dim=(16, 32), latent_dim=8)
    x = torch.randn(5, 10)
    out = encoder(x)
    assert out.shape == (5, 8), "Encoder output shape incorrect."


def test_decoder() -> None:
    """Test that the Decoder can be instantiated and run a forward pass."""
    decoder = Decoder(latent_dim=8, decoder_dim=(32, 16), data_dim=10)
    z = torch.randn(5, 8)
    out = decoder(z)
    assert out.shape == (5, 10), "Decoder output shape incorrect."


def test_vae() -> None:
    """Test that the VAE can be instantiated and run a full forward pass."""
    vae = VAE(input_dim=10, condition_dim=0, encoder_dim=(16, 32), latent_dim=8)
    x = torch.randn(5, 10)
    recon, mu, logvar = vae(x, None)
    assert recon.shape == (5, 10), "VAE recon output shape incorrect."
    assert mu.shape == (5, 8), "VAE mu output shape incorrect."
    assert logvar.shape == (5, 8), "VAE logvar output shape incorrect."


def test_discriminator() -> None:
    """Test that the Discriminator can be instantiated and run a forward pass."""
    disc = Discriminator(input_dim=10, discriminator_dim=(16, 8), pac=1)
    x = torch.randn(5, 10)
    out = disc(x)
    assert out.shape == (5, 1), "Discriminator output shape incorrect."


def test_generator() -> None:
    """Test that the Generator can be instantiated and run a forward pass."""
    decoder = Decoder(latent_dim=8, decoder_dim=(32, 16), data_dim=10)
    gen = Generator(decoder)
    z = torch.randn(5, 8)
    out = gen(z, None)
    assert out.shape == (5, 10), "Generator output shape incorrect."


@pytest.mark.parametrize("cuda", [False, True])
def test_vectgan_init(
    cuda: bool, mock_transformer: MagicMock, mock_data_sampler: MagicMock
) -> None:
    """Test that VectGan can be instantiated correctly and that correct initial values are set."""
    if cuda and not torch.cuda.is_available():
        pytest.skip("CUDA not available on this machine.")

    model = VectGan(latent_dim=8, discriminator_dim=(16, 8), epochs=1, batch_size=4, cuda=cuda)
    model._transformer = mock_transformer
    model._data_sampler = mock_data_sampler

    model.vae = VAE(
        model._transformer.output_dimensions,
        model._data_sampler.dim_cond_vec(),
        model._encoder_dim,
        model._latent_dim,
    ).to(model._device)
    model._generator = Generator(model.vae.decoder).to(model._device)
    model._discriminator = Discriminator(
        model._transformer.output_dimensions + model._data_sampler.dim_cond_vec(),
        model._discriminator_dim,
        pac=model.pac,
        lambda_=model.lambda_,
    ).to(model._device)

    assert model.vae is not None, "VAE not initialised."
    assert model._generator is not None, "Generator not initialised."
    assert model._discriminator is not None, "Discriminator not initialised."


def test_vectgan_fit(
    simple_data: np.ndarray, mock_transformer: MagicMock, mock_data_sampler: MagicMock
) -> None:
    """Test that VectGan's fit method runs a short training loop without errors."""
    mock_transformer.output_info_list = [{"dim": 10, "activation_fn": "tanh"}]
    mock_transformer.inverse_transform.side_effect = lambda x: x
    mock_data_sampler.dim_cond_vec.return_value = 0
    mock_data_sampler.sample_condvec.return_value = None
    mock_data_sampler.sample_original_condvec.return_value = None

    model = VectGan(latent_dim=8, discriminator_dim=(16, 8), epochs=1, batch_size=8, cuda=False)

    model._transformer = mock_transformer
    model._data_sampler = mock_data_sampler

    model.fit(simple_data)


def test_vectgan_sample(
    simple_data: np.ndarray, mock_transformer: MagicMock, mock_data_sampler: MagicMock
) -> None:
    """Test that sampling works after fitting."""
    mock_transformer.output_info_list = [{"dim": 10, "activation_fn": "tanh"}]
    mock_transformer.inverse_transform.side_effect = lambda x: x
    mock_data_sampler.dim_cond_vec.return_value = 0
    mock_data_sampler.sample_condvec.return_value = None
    mock_data_sampler.sample_original_condvec.return_value = None

    model = VectGan(latent_dim=8, discriminator_dim=(16, 8), epochs=1, batch_size=8, cuda=False)

    model._transformer = mock_transformer
    model._data_sampler = mock_data_sampler

    model.fit(simple_data)

    sampled = model.sample(10)
    assert sampled.shape == (10, 10), "Sampled data shape mismatch."
    assert isinstance(sampled, np.ndarray), "Sampled data not returned as numpy array."


def test_vectgan_save_load(
    simple_data: np.ndarray, mock_transformer: MagicMock, mock_data_sampler: MagicMock
) -> None:
    """Test that the model can be saved and loaded."""
    mock_transformer.output_info_list = [{"dim": 10, "activation_fn": "tanh"}]
    mock_transformer.inverse_transform.side_effect = lambda x: x
    mock_data_sampler.dim_cond_vec.return_value = 0
    mock_data_sampler.sample_condvec.return_value = None
    mock_data_sampler.sample_original_condvec.return_value = None

    model = VectGan(latent_dim=8, discriminator_dim=(16, 8), epochs=1, batch_size=8, cuda=False)

    model._transformer = mock_transformer
    model._data_sampler = mock_data_sampler
    model.fit(simple_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_model.pt")
        model.save(save_path)
        assert os.path.exists(save_path), "Model file was not saved."

        loaded_model = VectGan.load(save_path)
        assert loaded_model is not None, "Loaded model is None."
        sampled = loaded_model.sample(10)
        assert sampled.shape == (10, 10), "Loaded model sample shape mismatch."
