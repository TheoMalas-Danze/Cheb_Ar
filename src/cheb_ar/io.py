"""JSON (de)serialization helpers for sweep results.

Previously copy-pasted as ``_to_jsonable`` / ``_from_jsonable`` / ``load_json``
in every sweep script.
"""

import json

import jax
import numpy as np


def to_jsonable(obj):
    """Recursively convert jax/numpy arrays and complex numbers to JSON types.

    Complex values become ``{"real": ..., "imag": ...}``; arrays become nested
    lists; numpy/jax scalars become plain Python ``int``/``float``.
    """
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    # jax / numpy arrays -> numpy, then recurse on the python object
    if isinstance(obj, (jax.Array, np.ndarray)):
        return to_jsonable(np.asarray(obj).tolist())
    if isinstance(obj, (np.complexfloating, complex)):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def from_jsonable(obj):
    """Recursively convert JSON types back to numpy/complex objects.

    ``{"real": ..., "imag": ...}`` dicts become complex numbers;
    lists become numpy arrays if they contain numeric/complex values.
    """
    if isinstance(obj, dict):
        # Complex number encoding
        if set(obj.keys()) == {"real", "imag"}:
            return complex(obj["real"], obj["imag"])
        # Regular dict: recurse on values
        return {k: from_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        converted = [from_jsonable(v) for v in obj]
        # Convert to numpy array if all elements are numeric or complex
        if all(isinstance(v, (int, float, complex, np.ndarray)) for v in converted):
            return np.array(converted)
        return converted
    return obj  # int, float, str -> keep as-is


def load_json(path):
    """Load a JSON file and return a dict with numpy/complex values."""
    with open(path, "r") as f:
        raw = json.load(f)
    return from_jsonable(raw)
