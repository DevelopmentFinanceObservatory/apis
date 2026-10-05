"""Persistence errors. Connection and pool boundaries wrap driver exceptions here."""

from __future__ import annotations


class DatasetStoreError(Exception):
    """Base for every dataset_store exception."""


class DatasetStoreConfigError(DatasetStoreError):
    """Required connection settings are missing or invalid."""


class DatasetStoreConnectionError(DatasetStoreError):
    """Connection transport or authentication failed. The driver error is `__cause__`."""


class PoolTimeoutError(DatasetStoreError):
    """The borrow timeout expired before a connection became free.

    Connection dial failures raise `DatasetStoreConnectionError`.
    """
