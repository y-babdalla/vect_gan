"""BaseSynthesizer module.

Provides the base class for default synthesizers in CTGAN, along with
random state management decorators and context managers.
"""

import contextlib
from collections.abc import Callable, Generator

import numpy as np
import torch


@contextlib.contextmanager
def set_random_states(
    random_state: int | tuple[np.random.RandomState, torch.Generator],
    set_model_random_state: Callable[[tuple[np.random.RandomState, torch.Generator]], None],
) -> Generator[None, None, None]:
    """Context manager for managing the random state.

    This sets the numpy and torch random states for the duration of the context
    and reverts them afterward. It also calls `set_model_random_state` to store
    the current states in the model after the context completes.

    Args:
        random_state: Either an integer seed or a tuple containing
            (numpy.random.RandomState, torch.Generator).
        set_model_random_state: Function used to set the random state on the model.
    """
    original_np_state = np.random.get_state()
    original_torch_state = torch.get_rng_state()

    random_np_state, random_torch_state = random_state

    np.random.set_state(random_np_state.get_state())
    torch.set_rng_state(random_torch_state.get_state())

    try:
        yield
    finally:
        current_np_state = np.random.RandomState()
        current_np_state.set_state(np.random.get_state())
        current_torch_state = torch.Generator()
        current_torch_state.set_state(torch.get_rng_state())
        set_model_random_state((current_np_state, current_torch_state))

        np.random.set_state(original_np_state)
        torch.set_rng_state(original_torch_state)


def random_state_decorator(function: Callable) -> Callable:
    """Decorator to set the random state before calling the wrapped function.

    Checks if the class instance has a stored `random_states`. If present, it uses
    the context manager to set the states before execution.
    """

    def wrapper(self: "BaseSynthesizer", *args: object, **kwargs: object) -> object:
        """Wrapper function to set the random state before the original function call.

        Args:
            self: The instance of the class that contains this decorator.
            *args: Positional arguments to pass to the wrapped function.
            **kwargs: Keyword arguments to pass to the wrapped function.

        Returns:
            The result of calling the original function.
        """
        if self.random_states is None:
            return function(self, *args, **kwargs)

        with set_random_states(self.random_states, self.set_random_state):
            return function(self, *args, **kwargs)

    return wrapper


class BaseSynthesizer:
    """Base class for all default synthesizers of CTGAN.

    Provides methods for managing devices, saving/loading the model,
    and handling random states for reproducibility.
    """

    random_states: tuple[np.random.RandomState, torch.Generator] | None = None

    def __init__(self):
        self._device = None

    def __getstate__(self) -> dict:
        """Improve pickling state for ``BaseSynthesizer``.

        Convert to ``cpu`` device before starting the pickling process in order to be able to
        load the model even when used from an external tool such as ``SDV``. Also, if
        ``random_states`` are set, store their states as dictionaries rather than generators.

        Returns:
            dict:
                Python dict representing the object.
        """
        device_backup = self._device
        self.set_device(torch.device("cpu"))
        state = self.__dict__.copy()
        self.set_device(device_backup)
        if (
            isinstance(self.random_states, tuple)
            and isinstance(self.random_states[0], np.random.RandomState)
            and isinstance(self.random_states[1], torch.Generator)
        ):
            state["_numpy_random_state"] = self.random_states[0].get_state()
            state["_torch_random_state"] = self.random_states[1].get_state()
            state.pop("random_states")

        return state

    def __setstate__(self, state: dict) -> None:
        """Restore the state of a ``BaseSynthesizer``.

        Restore the ``random_states`` from the state dict if those are present and then
        set the device according to the current hardware.
        """
        if "_numpy_random_state" in state and "_torch_random_state" in state:
            np_state = state.pop("_numpy_random_state")
            torch_state = state.pop("_torch_random_state")

            current_torch_state = torch.Generator()
            current_torch_state.set_state(torch_state)

            current_numpy_state = np.random.RandomState()
            current_numpy_state.set_state(np_state)
            state["random_states"] = (current_numpy_state, current_torch_state)

        self.__dict__ = state
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.set_device(device)

    def save(self, path: str) -> None:
        """Save the model to the specified path using `torch.save`.

        Moves model to `cpu` prior to saving, then restores device afterward.

        Args:
            path: The file path to save the model to.
        """
        device_backup = self._device
        self.set_device(torch.device("cpu"))
        torch.save(self, path)
        self.set_device(device_backup)

    @classmethod
    def load(cls, path: str) -> "BaseSynthesizer":
        """Load the model stored in the specified path using `torch.load`.

        Moves the loaded model to the available device (GPU if available, otherwise CPU).

        Args:
            path: The file path from which to load the model.

        Returns:
            A `BaseSynthesizer` instance with the loaded model.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.load(path)
        model.set_device(device)
        return model

    def set_random_state(
        self, random_state: int | tuple[np.random.RandomState, torch.Generator] | None
    ) -> None:
        """Set the random state for both numpy and torch.

        Args:
            random_state: An integer seed or a tuple containing
                (numpy.random.RandomState, torch.Generator), or None to clear.
        """
        if random_state is None:
            self.random_states = random_state
        elif isinstance(random_state, int):
            self.random_states = (
                np.random.RandomState(seed=random_state),
                torch.Generator().manual_seed(random_state),
            )
        elif (
            isinstance(random_state, tuple)
            and isinstance(random_state[0], np.random.RandomState)
            and isinstance(random_state[1], torch.Generator)
        ):
            self.random_states = random_state
        else:
            raise TypeError(
                f"`random_state` {random_state} must be an int or a tuple of "
                "(np.random.RandomState, torch.Generator)."
            )

    def set_device(self, device: torch.device) -> None:
        """Set the device where the model should reside.

        Args:
            device: A `torch.device` specifying CPU or GPU.
        """
        self._device = device
