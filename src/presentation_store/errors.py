"""Configuration and store exceptions for `presentation_store`. Connection failures use
`dataset_store.errors.DatasetStoreConnectionError`.
"""

from __future__ import annotations


class PresentationStoreError(Exception):
    """Base exception for this package."""


class PresentationStoreConfigError(PresentationStoreError):
    """A required connection setting is missing or invalid."""
