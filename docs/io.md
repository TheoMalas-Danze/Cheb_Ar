# JSON (de)serialization helpers

Documentation for `src/cheb_ar/io.py` — previously copy-pasted as
`_to_jsonable` / `_from_jsonable` / `load_json` in every sweep script.

## Functions

- `to_jsonable(obj)` — recursive converter for writing sweep results:
  jax/numpy arrays → nested lists, complex numbers →
  `{"real": ..., "imag": ...}`, numpy scalars → plain Python `int`/`float`;
  dicts/lists/tuples are recursed into.
- `from_jsonable(obj)` — the inverse: `{"real", "imag"}` dicts → `complex`,
  lists whose elements are all numeric/complex → `numpy` arrays.
- `load_json(path)` — `json.load` + `from_jsonable`; what
  `sweep_alpha.py --warm-start-file` uses to recover the `x_ritz` vectors.

## Round-trip caveats

The round trip `from_jsonable(to_jsonable(x))` is faithful for the sweep
results but not for arbitrary objects:

- tuples come back as lists (or arrays, if numeric);
- *every* all-numeric list becomes an `ndarray`, even if it started as a
  plain Python list;
- a dict whose keys are exactly `{"real", "imag"}` is always decoded as a
  complex number — don't use that pair of keys for anything else in result
  dicts;
- dtypes are not preserved beyond the int/float/complex distinction (fine
  here: everything is double precision).
