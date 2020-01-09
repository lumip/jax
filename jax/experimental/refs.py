import jax

from jax import nn
from jax import numpy as jnp
from jax import random
from jax.util import partial

from collections import defaultdict
import threading

class Ref(threading.local):
  """A container for managing state in Python."""
  __slots__ = ["value"]
  def load(self):
    return self.value
  def store(self, value):
    self.value = value
  __init__ = store

# Three choices for Ref's pytree behavior:
# 1. register Ref as a pytree with its object identity as metadata
# 2. register Ref as a pytree that always throws an error
# 3. (chosen) don't register Ref as a pytree (treat it as a leaf)
#    --> enables using Refs in tree_util but not jit

def tree_load(ref_tree, typ=Ref):
  def load(ref):
    if isinstance(ref, typ):
      return ref.load()
  return jax.tree_map(load, ref_tree)

def tree_store(ref_tree, val_tree, typ=Ref):
  def store(ref, val):
    if isinstance(ref, typ):
      ref.store(val)
  jax.tree_multimap(store, ref_tree, val_tree)

def collect(fun, ref_tree):
  def inner(*args, **kwargs):
    out = fun(*args, **kwargs)
    val_tree = tree_load(ref_tree)
    return val_tree, out
  return inner

def inject(fun, ref_tree):
  def inner(val_tree, *args, **kwargs):
    tree_store(ref_tree, val_tree)
    return fun(*args, **kwargs)
  return inner

# maybe a function that does both

# examples

ref = Ref(None)
def foo(x):
  y = x ** 2
  ref.store(y)
  return y + 1

# tagging

_tags = defaultdict(lambda: Ref(None))
def tag(value, name):
  ref = _tags[name]
  ref.store(value)
  return ref.load()

# NNs

_global_PRNG_key = Ref(random.PRNGKey(0))
def next_key():
  key1, key2 = random.split(_global_PRNG_key.load())
  _global_PRNG_key.store(key1)
  return key2

class Parameter(Ref):
  """A trainable parameter."""
  pass

class Buffer(Ref):
  """A container for non-trainable state."""

class Module:
  def variables(self, typ=Ref):
    for v in self.__dict__.values():
      if isinstance(v, typ):
        yield v
      elif isinstance(v, Module):
        yield from v.variables()

def _module_to_tuple(module):
  refs = tuple(module.variables())
  for ref in refs:
    # avoid keeping unnecessary DeviceArray references in pytree metadata
    del ref.value
  return refs, module

def _module_from_tuple(module, tup):
  for module_ref, tuple_ref in zip(module.variables(), tup):
    module_ref.store(tuple_ref.load())
  return module

jax.tree_util.register_pytree_node(Module, _module_to_tuple, _module_from_tuple)

class Linear(Module):
  def __init__(self, nI, nO, bias=True,
               weight_init=nn.initializers.lecun_normal,
               bias_init=nn.initializers.zeros):
    self.W = Parameter(weight_init(next_key(), (nI, nO)))
    if bias:
      self.b = Parameter(bias_init(next_key(), (nO,)))
  def __call__(self, x):
    return x @ self.W.load() + self.b.load()

def init():
  layer1 = Linear(5, 10)
  layer2 = Linear(5, 10)
  layer1.W = layer2.W

  model = [layer1, layer2]
functional_init = inject(init, _global_PRNG_key)
model = functional_init(random.PRNGKey(1))

def loss(x):
  return layer2(layer1(x))

params = tree_load(model, Parameter)

functional_loss = partial(inject(loss, model), params)

def train_step(x):
  grads = jax.grad(functional_loss)(params, x)


