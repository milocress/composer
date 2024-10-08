# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Serialization interface used by checkpointing."""

from __future__ import annotations

from typing import Any

__all__ = ['Serializable']


class Serializable:
    """Interface for serialization; used by checkpointing."""

    def state_dict(self) -> dict[str, Any]:
        """Returns a dictionary representing the internal state.

        The returned dictionary must be pickale-able via :func:`torch.save`.

        Returns:
            dict[str, Any]: The state of the object.
        """
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restores the state of the object.

        Args:
            state (dict[str, Any]): The state of the object, as previously returned by :meth:`.state_dict`.
        """
        pass
