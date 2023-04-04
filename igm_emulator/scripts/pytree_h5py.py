"""Save/load pytrees to disk."""

import collections
import h5py
import jax
import numpy as np
import sys
import os
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator')
from haiku_custom_forward import _custom_forward_fn
import haiku as hk

def save(filepath, tree):
  """Saves a pytree to an hdf5 file.
  Args:
    filepath: str, Path of the hdf5 file to create.
    tree: pytree, Recursive collection of tuples, lists, dicts,
      namedtuples and numpy arrays to store.
  """
  with h5py.File(filepath, 'a') as f:
    _savetree(tree, f, 'best_params')
    f.close()


def load(filepath):
  """Loads a pytree from an hdf5 file.
  Args:
    filepath: str, Path of the hdf5 file to load.
  """
  with h5py.File(filepath, 'r') as f:
    return _loadtree(f['best_params'])
  f.close()

def _is_namedtuple(x):
  """Duck typing check if x is a namedtuple."""
  return isinstance(x, tuple) and getattr(x, '_fields', None) is not None


def _savetree(tree, f, name):
  """Recursively save a pytree to an h5 file group."""
  group = f.create_group(name)
  for row, module in enumerate(sorted(tree)):
    subgroup = group.create_group(module)
    subgroup.create_dataset('w', data=tree[module]['w'])
    subgroup.create_dataset('b', data=tree[module]['b'])


def _loadtree(tree):
  """Recursively load a pytree from an h5 file group."""
  #custom_forward = hk.without_apply_rng(hk.transform(_custom_forward_fn))
  #params = custom_forward.init(rng=42, x=np.ones([10,1]))
  params = {}
  for row, module in enumerate(sorted(tree)):
      print(tree[module].keys())
      p = {dict([f'{module}', dict([('w',tree[module]['w']),('b',tree[module]['b'])])])}
      params = params.update(p)
  return params