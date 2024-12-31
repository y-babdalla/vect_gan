"""VectGan is a Variationally Encoded Conditional Tabular GAN.

This module contains the main classes, modules, and functions
to run and train VECT GAN for tabular data generation.
"""

import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vect_gan.synthesizers.base import BaseSynthesizer, random_state_decorator
from vect_gan.utils.data_sampler import DataSampler
from vect_gan.utils.data_transformer import DataTransformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Residual(nn.Module):
    """Residual layer for the CTGAN.

    This class implements a feed-forward residual layer with a skip
    connection between input and output.
    """

    def __init__(self, i: int, o: int) -> None:
        """Initialise the Residual layer."""
        super().__init__()
        self.fc = nn.Linear(i, o)
        self.bn = nn.BatchNorm1d(o)
        self.relu = nn.ReLU()

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Apply the Residual layer to the input tensor."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Encoder(nn.Module):
    """Encoder for the VAE within CTGAN."""

    def __init__(self, input_dim: int, encoder_dim: tuple[int, ...], latent_dim: int) -> None:
        """Initialise the Encoder.

        Args:
            input_dim: Dimensionality of the encoder's input.
            encoder_dim: Sequence of dimensions for each Residual layer.
            latent_dim: Dimensionality of the latent space.
        """
        super().__init__()
        dim = input_dim
        seq: list[nn.Module] = []

        for item in encoder_dim:
            seq.append(Residual(dim, item))
            dim += item

        self.final_layer = nn.Linear(dim, latent_dim)
        self.seq = nn.Sequential(*seq)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Return the latent representation from the input."""
        h = self.seq(input_)
        return self.final_layer(h)


class Decoder(nn.Module):
    """Decoder for the VAE within CTGAN."""

    def __init__(self, latent_dim: int, decoder_dim: tuple[int, ...], data_dim: int) -> None:
        """Initialise the Decoder.

        Args:
            latent_dim: Dimensionality of the latent space.
            decoder_dim: Sequence of dimensions for each Residual layer.
            data_dim: Dimensionality of the final output.
        """
        super().__init__()
        dim = latent_dim
        seq: list[nn.Module] = []

        for item in decoder_dim:
            seq.append(Residual(dim, item))
            dim += item

        seq.append(nn.Linear(dim, data_dim))
        self.seq = nn.Sequential(*seq)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Return the reconstructed data from the input latent vector."""
        return self.seq(input_)


class VAE(nn.Module):
    """Variational Autoencoder component of the CTGAN."""

    def __init__(
        self, input_dim: int, condition_dim: int, encoder_dim: tuple[int, ...], latent_dim: int
    ) -> None:
        """Initialise the VAE.

        Args:
            input_dim: Dimensionality of the input data.
            condition_dim: Dimensionality of the conditional vector.
            encoder_dim: Sequence of dimensions for each Residual layer in the encoder.
            latent_dim: Dimensionality of the latent space.
        """
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = condition_dim
        self.latent_dim = latent_dim
        self.encoder_dim = encoder_dim

        self.encoder = Encoder(self.input_dim + self.cond_dim, self.encoder_dim, self.latent_dim)

        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)

        self.decoder = Decoder(
            self.latent_dim + self.cond_dim, self.encoder_dim[::-1], self.input_dim
        )

    def encode(self, x: torch.Tensor, c: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode x (and condition) into mu and logvar."""
        if c is not None:
            x = torch.cat([x, c], dim=1)
        x = x.to(torch.float32)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Perform the reparameterisation trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c: torch.Tensor | None) -> torch.Tensor:
        """Decode latent vector z (and condition) into reconstructed data."""
        if c is not None:
            z = torch.cat([z, c], dim=1)
        z = z.to(torch.float32)
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the encoder, reparameterise, then decode."""
        mu, logvar = self.encode(x, c)
        z = self.reparameterise(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar


class Discriminator(nn.Module):
    """Discriminator for the CTGAN."""

    def __init__(
        self,
        input_dim: int,
        discriminator_dim: tuple[int, ...],
        pac: int = 10,
        lambda_: float = 10.0,
    ) -> None:
        """Initialise the Discriminator.

        Args:
            input_dim: Dimensionality of the input.
            discriminator_dim: Sequence of linear layer sizes.
            pac: Number of samples to group together.
            lambda_: Weight for the gradient penalty.
        """
        super().__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        self.lambda_ = lambda_
        seq: list[nn.Module] = []

        for item in list(discriminator_dim):
            seq += [nn.Linear(dim, item), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
            dim = item

        seq += [nn.Linear(dim, 1)]
        self.seq = nn.Sequential(*seq)

    def calc_gradient_penalty(
        self,
        real_data: torch.Tensor,
        fake_data_list: list[torch.Tensor],
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Compute the gradient penalty."""
        gradient_penalty = 0.0
        for fake_data in fake_data_list:
            alpha = torch.rand(real_data.size(0) // self.pac, 1, 1, device=device)
            alpha = alpha.repeat(1, self.pac, real_data.size(1))
            alpha = alpha.view(-1, real_data.size(1))

            interpolates = alpha * real_data + ((1 - alpha) * fake_data)
            disc_interpolates = self(interpolates)

            gradients = torch.autograd.grad(
                outputs=disc_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            # Norm across the concatenated dimension
            gradients_view = gradients.view(-1, self.pac * real_data.size(1)).norm(2, dim=1) - 1
            gradient_penalty += (gradients_view**2).mean() * self.lambda_

        return gradient_penalty / len(fake_data_list)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Apply the Discriminator to the input."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


class Generator(nn.Module):
    """Generator class for the CTGAN."""

    def __init__(self, decoder: Decoder) -> None:
        """Initialise the Generator with a Decoder."""
        super().__init__()
        self.decoder = decoder

    def forward(self, input_: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        """Generate fake data from input and optional condition."""
        if condition is not None:
            input_ = torch.cat([input_, condition], dim=1)
        input_ = input_.to(torch.float32)
        return self.decoder(input_)


class VectGan(BaseSynthesizer):
    """Conditional Table GAN Synthesizer with a VAE extension.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        latent_dim (int):
            Size of the random sample passed to the Generator. Defaults to 64.
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers.
            A Linear Layer will be created for each one of the values provided.
            Defaults to (32, 32, 32).
        generator_lr (float):
            Learning rate for the generator. Defaults to 1e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-5.
        batch_size (int):
            Number of data samples to process in each step. Default is 32.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            Defaults to 2.
        log_frequency (bool):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to True.
        verbose (bool):
            Whether to have print statements for progress results. Defaults to True.
        epochs (int):
            Number of training epochs. Default is 100.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 4.
        cuda (bool or str):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to True.
        kl_weight (float):
            Weight for the Kullback-Leibler divergence in the VAE. Defaults to 1.0.
        encoder_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals in the Encoder.
            A Residual Layer will be created for each one of the values provided.
            Defaults to (256, 512).
        encoder_lr (float):
            Learning rate for the encoder. Defaults to 1e-5.
        encoder_decay (float):
            Encoder weight decay for the Adam Optimizer. Defaults to 2e-6.
        vae_weight (float):
            Weight for the VAE loss. Defaults to 1.0.
        cond_loss_weight (float):
            Weight for the conditional loss. Defaults to 1.0.
        lambda_ (float):
            Weight for the gradient penalty. Defaults to 10.0.
        enforce_min_max_values (bool):
            Whether to enforce the minimum and maximum values of the original data.
            Defaults to True.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        discriminator_dim: tuple[int, ...] = (32, 32, 32),
        generator_lr: float = 1e-4,
        generator_decay: float = 1e-6,
        discriminator_lr: float = 2e-4,
        discriminator_decay: float = 1e-5,
        batch_size: int = 32,
        discriminator_steps: int = 2,
        log_frequency: bool = True,
        verbose: bool = True,
        epochs: int = 100,
        pac: int = 4,
        cuda: bool | str = True,
        kl_weight: float = 1.0,
        encoder_dim: tuple[int, ...] = (256, 512),
        encoder_lr: float = 1e-5,
        encoder_decay: float = 2e-6,
        vae_weight: float = 1.0,
        cond_loss_weight: float = 1.0,
        lambda_: float = 10.0,
        enforce_min_max_values: bool = True,
    ) -> None:
        """Initialise the VectGan model.

        Args:
            latent_dim: Size of the random sample passed to the Generator.
            discriminator_dim: Sequence of linear layer sizes for the Discriminator.
            generator_lr: Learning rate for the Generator.
            generator_decay: Weight decay for the Generator optimiser.
            discriminator_lr: Learning rate for the Discriminator.
            discriminator_decay: Weight decay for the Discriminator optimiser.
            batch_size: Number of data samples to process in each step.
            discriminator_steps: Number of Discriminator updates per Generator update.
            log_frequency: Whether to use log frequency in conditional sampling.
            verbose: Whether to print progress.
            epochs: Number of training epochs.
            pac: Number of samples to group together in the Discriminator.
            cuda: Whether (or which GPU) to use for computation.
            kl_weight: Weight for the KL divergence in the VAE.
            encoder_dim: Sequence of dimensions for Residual layers in the VAE encoder.
            encoder_lr: Learning rate for the VAE encoder.
            encoder_decay: Weight decay for the VAE encoder optimiser.
            vae_weight: Weight for the VAE loss.
            cond_loss_weight: Weight for the conditional loss.
            lambda_: Weight for the gradient penalty.
            enforce_min_max_values: Whether to enforce min/max values in generation.
        """
        super().__init__()
        assert batch_size % 2 == 0

        torch.set_default_dtype(torch.float32)

        self.vae: VAE | None = None
        self._discriminator: Discriminator | None = None
        self._train_data: torch.Tensor | None = None

        self.optimizerE: optim.Adam | None = None
        self.optimizerG: optim.Adam | None = None
        self.optimizerD: optim.Adam | None = None

        self._latent_dim = latent_dim
        self._discriminator_dim = discriminator_dim
        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay
        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac
        self._kl_weight = float(kl_weight)
        self._encoder_dim = encoder_dim
        self.encoder_decay = encoder_decay
        self.cond_loss_weight = cond_loss_weight
        self.lambda_ = lambda_

        if not cuda or not torch.cuda.is_available():
            device = "cpu"
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = "cuda"

        self._device = torch.device(device)
        self._transformer: DataTransformer | None = None
        self._data_sampler: DataSampler | None = None
        self._generator: Generator | None = None
        self.loss_values: pd.DataFrame | None = None
        self._encoder_lr = encoder_lr
        self.vae_weight = vae_weight
        self._enforce_min_max_values = enforce_min_max_values
        self._min_max_values: dict[str, tuple[float, float]] = {}
        self.writer = SummaryWriter()

    @staticmethod
    def _gumbel_softmax(
        logits: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
        eps: float = 1e-10,
        dim: int = -1,
    ) -> torch.Tensor:
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        for _ in range(10):
            transformed = nn.functional.gumbel_softmax(
                logits, tau=tau, hard=hard, eps=eps, dim=dim
            )
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError("gumbel_softmax returning NaN.")

    def _apply_activate(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the correct activation function to Generator outputs."""
        data_t: list[torch.Tensor] = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == "tanh":
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == "softmax":
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f"Unexpected activation function {span_info.activation_fn}.")
        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data: torch.Tensor, c: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy on discrete columns for conditional fidelity."""
        loss_list: list[torch.Tensor] = []
        st = 0
        st_c = 0

        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != "softmax":
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = nn.functional.cross_entropy(
                        data[:, st:ed], torch.argmax(c[:, st_c:ed_c], dim=1), reduction="none"
                    )
                    loss_list.append(tmp)
                    st = ed
                    st_c = ed_c

        stacked_loss = torch.stack(loss_list, dim=1)
        return (stacked_loss * m).sum() / data.size()[0]

    def _reconstruction_loss(self, recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss for discrete and continuous columns."""
        st = 0
        recon_loss = 0.0

        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                ed = st + span_info.dim
                if span_info.activation_fn == "softmax":
                    target = torch.argmax(x[:, st:ed], dim=1)
                    prediction = recon_x[:, st:ed]
                    recon_loss += nn.functional.cross_entropy(prediction, target)
                elif span_info.activation_fn == "tanh":
                    recon_loss += nn.functional.mse_loss(
                        torch.tanh(recon_x[:, st:ed]), x[:, st:ed]
                    )
                else:
                    raise ValueError(f"Unknown activation function {span_info.activation_fn}")
                st = ed
        return recon_loss

    @staticmethod
    def _validate_discrete_columns(
        train_data: pd.DataFrame | np.ndarray, discrete_columns: list[int | str]
    ) -> None:
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError("train_data must be pd.DataFrame or np.ndarray.")

        if invalid_columns:
            raise ValueError(f"Invalid columns found: {invalid_columns}")

    @random_state_decorator
    def fit(  # noqa: C901, PLR0912
        self,
        train_data: pd.DataFrame | np.ndarray,
        discrete_columns: list[int | str] = (),
        fine_tuning: bool = False,
    ) -> None:
        """Fit the VAE-CTGAN to the training data."""
        self._validate_discrete_columns(train_data, discrete_columns)

        if not fine_tuning:
            self._transformer = DataTransformer()
            self._transformer.fit(train_data, discrete_columns)
            self._train_data = self._transformer.transform(train_data)
            self._data_sampler = DataSampler(
                self._train_data, self._transformer.output_info_list, self._log_frequency
            )

        self.vae = VAE(
            self._transformer.output_dimensions,
            self._data_sampler.dim_cond_vec(),
            self._encoder_dim,
            self._latent_dim,
        ).to(self._device)

        self._generator = Generator(self.vae.decoder).to(self._device)

        self._discriminator = Discriminator(
            input_dim=self._transformer.output_dimensions + self._data_sampler.dim_cond_vec(),
            discriminator_dim=self._discriminator_dim,
            pac=self.pac,
            lambda_=self.lambda_,
        ).to(self._device)

        self.optimizerE = optim.Adam(
            self.vae.encoder.parameters(), lr=self._encoder_lr, weight_decay=self.encoder_decay
        )
        self.optimizerG = optim.Adam(
            self.vae.decoder.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )
        self.optimizerD = optim.Adam(
            self._discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        self.loss_values = pd.DataFrame(
            columns=["Epoch", "Generator Loss", "VAE Loss", "Discriminator Loss"]
        )

        epoch_iterator = tqdm(range(self._epochs), disable=(not self._verbose))
        if self._verbose:
            description = "Gen. ({gen:.2f}) | VAE ({vae:.2f}) | Discrim. ({dis:.2f})"
            epoch_iterator.set_description(description.format(gen=0, vae=0, dis=0))

        for epoch in epoch_iterator:
            data_loader = DataLoader(
                self._train_data, batch_size=self._batch_size, shuffle=True, drop_last=True
            )

            for _, batch in enumerate(data_loader):
                batch = batch.to(self._device).to(torch.float32)  # noqa: PLW2901
                condvec = self._data_sampler.sample_condvec(batch.shape[0])
                if condvec is not None:
                    c_real = torch.from_numpy(condvec[0]).to(self._device).float()
                else:
                    c_real = None

                # Discriminator training
                for _ in range(self._discriminator_steps):
                    self.optimizerD.zero_grad()

                    if c_real is not None:
                        real_cat = torch.cat([batch, c_real], dim=1).float()
                    else:
                        real_cat = batch

                    y_real = self._discriminator(real_cat)

                    recon_batch, mu, logvar = self.vae(batch, c_real)
                    recon_act = self._apply_activate(recon_batch)

                    if c_real is not None:
                        recon_cat = torch.cat([recon_act, c_real], dim=1).float()
                    else:
                        recon_cat = recon_act

                    y_recon = self._discriminator(recon_cat)

                    fakez = torch.randn(batch.shape[0], self._latent_dim, device=self._device)
                    if condvec is not None:
                        c_fake = torch.from_numpy(condvec[0]).to(self._device).float()
                    else:
                        c_fake = None

                    fake = self._generator(fakez, c_fake)
                    fake_act = self._apply_activate(fake)

                    if c_fake is not None:
                        fake_cat = torch.cat([fake_act, c_fake], dim=1).float()
                    else:
                        fake_cat = fake_act

                    y_fake = self._discriminator(fake_cat)

                    grad_pen = self._discriminator.calc_gradient_penalty(
                        real_cat, [fake_cat, recon_cat], self._device
                    )

                    # Compute discriminator loss
                    loss_d = (
                        -torch.mean(y_real)
                        + 0.5 * (torch.mean(y_fake) + torch.mean(y_recon))
                        + grad_pen
                    )

                    # Backward and optimize discriminator
                    loss_d.backward()
                    self.optimizerD.step()

                # Generator and VAE training
                self.optimizerE.zero_grad()
                self.optimizerG.zero_grad()

                # VAE forward pass
                recon_batch, mu, logvar = self.vae(batch, c_real)
                recon_act = self._apply_activate(recon_batch)

                # Generator adversarial loss on reconstructed data
                if c_real is not None:
                    recon_cat = torch.cat([recon_act, c_real], dim=1).float()
                else:
                    recon_cat = recon_act

                y_recon = self._discriminator(recon_cat)
                adv_loss_recon = -torch.mean(y_recon)

                # Generator adversarial loss on fake data
                fakez = torch.randn(batch.shape[0], self._latent_dim, device=self._device)
                if condvec is not None:
                    c_fake = torch.from_numpy(condvec[0]).to(self._device).float()
                else:
                    c_fake = None

                fake = self._generator(fakez, c_fake)
                fake_act = self._apply_activate(fake)

                if c_fake is not None:
                    fake_cat = torch.cat([fake_act, c_fake], dim=1).float()
                else:
                    fake_cat = fake_act

                y_fake = self._discriminator(fake_cat)
                adv_loss_fake = -torch.mean(y_fake)

                # Total adversarial loss
                gen_adversarial_loss = 0.5 * (adv_loss_fake + adv_loss_recon)

                if condvec is not None:
                    c = torch.from_numpy(condvec[0]).to(self._device).float()
                    m = torch.from_numpy(condvec[1]).to(self._device).float()
                    cond_loss_fake = self._cond_loss(fake, c, m)
                    cond_loss_real = self._cond_loss(recon_batch, c, m)
                    cond_loss = 0.5 * (cond_loss_real + cond_loss_fake)
                else:
                    cond_loss = 0

                vae_loss_value = (
                    self.vae_loss(recon_batch, batch, mu, logvar.float())
                    + cond_loss * self.cond_loss_weight
                )
                total_gen_loss = gen_adversarial_loss + vae_loss_value * self.vae_weight

                total_gen_loss.backward()
                self.optimizerE.step()
                self.optimizerG.step()

            generator_loss = gen_adversarial_loss.detach().cpu().item()
            vae_loss_cpu = vae_loss_value.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()

            self.writer.add_scalar("Loss/Generator", generator_loss, epoch)
            self.writer.add_scalar("Loss/VAE", vae_loss_cpu, epoch)
            self.writer.add_scalar("Loss/Discriminator", discriminator_loss, epoch)

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(
                        gen=generator_loss, vae=vae_loss_cpu, dis=discriminator_loss
                    )
                )

    def vae_loss(
        self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Compute the full VAE loss (reconstruction + KL)."""
        recon_loss = self._reconstruction_loss(recon_x, x) / x.size(0)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + self._kl_weight * kld

    @random_state_decorator
    def sample(
        self, n: int, condition_column: str | None = None, condition_value: str | None = None
    ) -> pd.DataFrame | np.ndarray:
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value
            )
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size
            )
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []

        for _ in range(steps):
            fakez = torch.randn(self._batch_size, self._latent_dim, device=self._device)
            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            c1 = None if condvec is None else torch.from_numpy(condvec).to(self._device)

            fake = self._generator(fakez, c1)
            fake_act = self._apply_activate(fake)
            data.append(fake_act.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        return self._transformer.inverse_transform(data)

    def set_device(self, device: str) -> None:
        """Set the device to be used ('GPU' or 'CPU')."""
        self._device = torch.device(device)
        if self._generator is not None:
            self._generator.to(self._device)

    def save(self, path: str) -> None:
        """Save the model to the specified path using torch.save."""
        state = {
            "model_parameters": {
                "latent_dim": self._latent_dim,
                "discriminator_dim": self._discriminator_dim,
                "generator_lr": self._generator_lr,
                "generator_decay": self._generator_decay,
                "discriminator_lr": self._discriminator_lr,
                "discriminator_decay": self._discriminator_decay,
                "batch_size": self._batch_size,
                "discriminator_steps": self._discriminator_steps,
                "log_frequency": self._log_frequency,
                "verbose": self._verbose,
                "epochs": self._epochs,
                "pac": self.pac,
                "cuda": str(self._device),
                "kl_weight": self._kl_weight,
                "encoder_dim": self._encoder_dim,
                "encoder_lr": self._encoder_lr,
                "encoder_decay": self.encoder_decay,
                "vae_weight": self.vae_weight,
                "cond_loss_weight": self.cond_loss_weight,
                "lambda_": self.lambda_,
            },
            "transformer": self._transformer,
            "data_sampler": self._data_sampler,
            "output_dimensions": self._transformer.output_dimensions,
            "dim_cond_vec": self._data_sampler.dim_cond_vec(),
            "vae_state_dict": self.vae.state_dict(),
            "generator_state_dict": self._generator.state_dict(),
            "discriminator_state_dict": self._discriminator.state_dict(),
            "optimizerE_state_dict": self.optimizerE.state_dict(),
            "optimizerG_state_dict": self.optimizerG.state_dict(),
            "optimizerD_state_dict": self.optimizerD.state_dict(),
            "writer": None,
        }
        torch.save(state, path)

    @classmethod
    def load(
        cls, path: str | None = None, skip_transformer_and_sampler: bool = False
    ) -> "VectGan":
        """Load the model from the specified path using torch.load.

        Args:
            path (str): The file path from which to load the model.
            skip_transformer_and_sampler (bool): Whether to skip loading the transformer and
                data sampler. Defaults to False.

        Returns:
            VectGan: An instance of VAE_CTGAN with loaded parameters.
        """
        if not path:
            path = (
                f"{os.path.dirname(os.path.abspath(__file__))}"
                "/../checkpoints/pre_trained_vect_gan.pt"
            )

        # Load the state dictionary
        state = torch.load(
            path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            weights_only=False,
        )

        # Extract model parameters
        params = state["model_parameters"]

        # Create a new instance with the saved parameters
        model = cls(**params)

        # Load transformer and data_sampler
        if not skip_transformer_and_sampler:
            model._transformer = state["transformer"]
            model._data_sampler = state["data_sampler"]
        else:
            model._transformer = None
            model._data_sampler = None

        output_dimensions = state["output_dimensions"]
        dim_cond_vec = state["dim_cond_vec"]

        # Initialise the VAE, generator, and discriminator
        model.vae = VAE(output_dimensions, dim_cond_vec, model._encoder_dim, model._latent_dim).to(
            model._device
        )

        model._generator = Generator(model.vae.decoder).to(model._device)

        model._discriminator = Discriminator(
            output_dimensions + dim_cond_vec, model._discriminator_dim, pac=model.pac
        ).to(model._device)

        # Load state dictionaries
        model.vae.load_state_dict(state["vae_state_dict"])
        model._generator.load_state_dict(state["generator_state_dict"])
        model._discriminator.load_state_dict(state["discriminator_state_dict"])

        # Initialise optimizers
        model.optimizerE = optim.Adam(
            model.vae.encoder.parameters(), lr=model._encoder_lr, weight_decay=model.encoder_decay
        )
        model.optimizerG = optim.Adam(
            model.vae.decoder.parameters(),
            lr=model._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=model._generator_decay,
        )
        model.optimizerD = optim.Adam(
            model._discriminator.parameters(),
            lr=model._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=model._discriminator_decay,
        )

        # Load optimiser states
        model.optimizerE.load_state_dict(state["optimizerE_state_dict"])
        model.optimizerG.load_state_dict(state["optimizerG_state_dict"])
        model.optimizerD.load_state_dict(state["optimizerD_state_dict"])

        return model

    @classmethod
    def fine_tune(  # noqa: C901, PLR0912
        cls,
        new_data: np.ndarray | pd.DataFrame | None = None,
        pre_trained_model_path: str | None = None,
        discrete_columns: list[int | str] = (),
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
        encoder_lr: float | None = None,
        generator_lr: float | None = None,
        discriminator_lr: float | None = None,
        encoder_decay: float | None = None,
        generator_decay: float | None = None,
        discriminator_decay: float | None = None,
        kl_weight: float | None = None,
        vae_weight: float | None = None,
        cond_loss_weight: float | None = None,
        discriminator_steps: int | None = None,
        lambda_: float | None = None,
        pac: int = 8,
    ) -> "VectGan":
        """Fine-tune a pre-trained VectGan model on new data.

        Fine-tuning works by loading the pre-trained model weights, however, the data transformer
        and data samplers need to be fit on your data first.

        Args:
            new_data (numpy.ndarray or pandas.DataFrame): The new data to fine-tune the model on.
            pre_trained_model_path (str): The file path from which to load the pre-trained model.
            discrete_columns (list-like): List of discrete columns to be used to generate the
                Conditional Vector. If ``train_data`` is a Numpy array, this list should contain
                the integer indices of the columns. Otherwise, if it is a ``pandas.DataFrame``,
                this list should contain the column names.
            epochs (int): Number of training epochs. Default is 100.
            batch_size (int): Number of data samples to process in each step. Default is 32.
            verbose (bool): Whether to have print statements for progress results.
                Defaults to True.
            encoder_lr (float): Learning rate for the encoder. Defaults to None.
            generator_lr (float): Learning rate for the generator. Defaults to None.
            discriminator_lr (float): Learning rate for the discriminator. Defaults to None.
            encoder_decay (float): Encoder weight decay for the Adam Optimizer. Defaults to None.
            generator_decay (float): Generator weight decay for the Adam Optimizer. Defaults to
                None.
            discriminator_decay (float): Discriminator weight decay for the Adam Optimizer.
                Defaults to None.
            kl_weight (float): Weight for the Kullback-Leibler divergence in the VAE. Defaults
                to None.
            vae_weight (float): Weight for the VAE loss. Defaults to None.
            cond_loss_weight (float): Weight for the conditional loss. Defaults to None.
            discriminator_steps (int): Number of discriminator updates to do for each generator
                update. Defaults to None.
            lambda_ (float): Weight for the gradient penalty. Defaults to None.
            pac (int): Number of samples to group together when applying the discriminator.
                Defaults to None.

        Returns:
            VectGan: A fine-tuned instance of VectGan.
        """
        # Load the pre-trained model
        pretrained_model = cls.load(pre_trained_model_path, skip_transformer_and_sampler=True)
        pretrained_model._verbose = verbose

        # Update parameters where relevant
        if encoder_lr is not None:
            pretrained_model._encoder_lr = encoder_lr
        if generator_lr is not None:
            pretrained_model._generator_lr = generator_lr
        if discriminator_lr is not None:
            pretrained_model._discriminator_lr = discriminator_lr
        if encoder_decay is not None:
            pretrained_model.encoder_decay = encoder_decay
        if generator_decay is not None:
            pretrained_model._generator_decay = generator_decay
        if discriminator_decay is not None:
            pretrained_model._discriminator_decay = discriminator_decay
        if kl_weight is not None:
            pretrained_model._kl_weight = kl_weight
        if vae_weight is not None:
            pretrained_model.vae_weight = vae_weight
        if cond_loss_weight is not None:
            pretrained_model.cond_loss_weight = cond_loss_weight
        if discriminator_steps is not None:
            pretrained_model._discriminator_steps = discriminator_steps
        if lambda_ is not None:
            pretrained_model.lambda_ = lambda_
        if epochs is not None:
            pretrained_model._epochs = epochs
        if batch_size is not None:
            pretrained_model._batch_size = batch_size
        if pac is not None:
            pretrained_model.pac = pac

        new_transformer = DataTransformer()
        new_transformer.fit(new_data, discrete_columns)
        transformed_new_data = new_transformer.transform(new_data)
        new_output_dim = new_transformer.output_dimensions
        pretrained_model._transformer = new_transformer
        pretrained_model._train_data = transformed_new_data

        pretrained_model._data_sampler = DataSampler(
            transformed_new_data, new_transformer.output_info_list, pretrained_model._log_frequency
        )

        cond_dim = pretrained_model._data_sampler.dim_cond_vec()

        new_vae = VAE(
            input_dim=new_output_dim,
            condition_dim=cond_dim,
            encoder_dim=pretrained_model._encoder_dim,
            latent_dim=pretrained_model._latent_dim,
        ).to(pretrained_model._device)

        cls._transfer_state_dict(
            new_vae.encoder, pretrained_model.vae.encoder.state_dict(), verbose
        )
        cls._transfer_state_dict(
            new_vae.decoder, pretrained_model.vae.decoder.state_dict(), verbose
        )

        pretrained_model.vae = new_vae
        pretrained_model._generator = Generator(new_vae.decoder).to(pretrained_model._device)

        new_discriminator = Discriminator(
            input_dim=new_output_dim + cond_dim,
            discriminator_dim=pretrained_model._discriminator_dim,
            pac=pretrained_model.pac,
            lambda_=pretrained_model.lambda_,
        ).to(pretrained_model._device)

        cls._transfer_state_dict(
            new_discriminator, pretrained_model._discriminator.state_dict(), verbose
        )
        pretrained_model._discriminator = new_discriminator

        pretrained_model.optimizerE = optim.Adam(
            pretrained_model.vae.encoder.parameters(),
            lr=pretrained_model._encoder_lr,
            weight_decay=pretrained_model.encoder_decay,
        )
        pretrained_model.optimizerG = optim.Adam(
            pretrained_model.vae.decoder.parameters(),
            lr=pretrained_model._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=pretrained_model._generator_decay,
        )
        pretrained_model.optimizerD = optim.Adam(
            pretrained_model._discriminator.parameters(),
            lr=pretrained_model._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=pretrained_model._discriminator_decay,
        )

        pretrained_model.fit(new_data, discrete_columns, fine_tuning=True)
        return pretrained_model

    @staticmethod
    def _transfer_state_dict(
        new_model: nn.Module, state_dict: dict[str, torch.Tensor], verbose: bool = False
    ) -> None:
        """Transfers weights from state_dict to new_model where possible.

        Args:
            new_model (torch.nn.Module): The new model instance.
            state_dict (dict): State dictionary from the pre-trained model.
            verbose (bool): Whether to print warnings about mismatches.

        """
        new_state_dict = new_model.state_dict()
        for name, param in state_dict.items():
            if name in new_state_dict:
                if new_state_dict[name].shape == param.shape:
                    new_state_dict[name].copy_(param)
                elif verbose:
                    logger.warning(f"Skipping parameter '{name}' due to shape mismatch.")
            elif verbose:
                logger.warning(f"Parameter '{name}' not found in new model.")
        new_model.load_state_dict(new_state_dict, strict=False)
