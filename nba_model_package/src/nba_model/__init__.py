"""
Betsync NBA model bundle.

The :mod:`nba_model` package exposes helper utilities for loading the trained
NBA hybrid model and running inference against prepared feature matrices.

Typical usage::

    from nba_model.inference import load_bundle

    bundle = load_bundle()
    preds = bundle.predict(features_df)
"""

from .inference import NBAModelBundle, load_bundle

__all__ = ["NBAModelBundle", "load_bundle"]

