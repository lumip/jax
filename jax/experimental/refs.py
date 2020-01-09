import jax

from jax import nn
from jax import numpy as jnp
from jax import random
from jax.util import partial

import threading

class Ref(threading.local):
  """A container for managing state in Python."""
  __slots__ = ["value"]
  def __init__(self, value):
    super().__init__()
    self.value = value
  def __repr__(self):
    return self.__class__.__name__ + '(' + repr(self.value) + ')'
  def load(self):
    return self.value
  def store(self, value):
    self.value = value
  def swap(self, value):
    value, self.value = self.value, value
    return value

# Three choices for Ref's pytree behavior:
# 1. register Ref as a pytree with its object identity as metadata
# 2. register Ref as a pytree that always throws an error
# 3. (chosen) don't register Ref as a pytree (treat it as a leaf)
#    --> enables using Refs in tree_util but not jit

def tree_load(ref_tree, typ=Ref):
  loaded = set()
  def load(ref):
    if isinstance(ref, typ) and ref not in loaded:
      loaded.add(ref)
      return ref.load()
  return jax.tree_map(load, ref_tree)

def tree_store(ref_tree, val_tree, typ=Ref):
  stored = set()
  def store(ref, val):
    if isinstance(ref, typ) and ref not in stored:
      stored.add(ref)
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

_tags = dict()
def tag(value, name):
  if name in _tags:
    return _tags[name].swap(value)
  else:
    _tags[name] = Ref(value)
    return value

# NNs

_global_PRNG_key = Ref(random.PRNGKey(0))
def next_key():
  key1, key2 = random.split(_global_PRNG_key.load())
  _global_PRNG_key.store(key1)
  return key2

class Parameter(Ref):
  """A trainable parameter."""

class Buffer(Ref):
  """A container for non-trainable state."""


class _ModuleMeta(type):
  def __init__(cls, name, bases, attrs):
    super(_ModuleMeta, cls).__init__(name, bases, attrs)
    def from_kv(keys, values):
      module = cls.__new__(cls)
      module.__dict__.update(**dict(zip(keys, values)))
      return module
    jax.tree_util.register_pytree_node(
        cls,
        lambda m: (list(m.__dict__.values()), list(m.__dict__.keys())),
        from_kv)

class Module(metaclass=_ModuleMeta):
  def __repr__(self):
    s = ', '.join(k + '=' + repr(v) for k, v in self.__dict__.items())
    return self.__class__.__name__ + '(' + s + ')'
  # def variables(self, typ=Ref):
  #   def inner(v):
  #     if isinstance(v, Module):
  #       yield from v.variables(typ)
  #     else:
  #       yield v
  #   for v in self.__dict__.values():
  #     if isinstance(v, Module):
  #       yield from v.variables(typ)
  #     else:
  #       for sub in jax.tree_flatten(v)[0]:
  #         yield from inner(sub)


class Linear(Module):
  def __init__(self, nI, nO, bias=True,
               weight_init=nn.initializers.lecun_normal(),
               bias_init=nn.initializers.zeros):
    self.W = Parameter(weight_init(next_key(), (nI, nO)))
    if bias:
      self.b = Parameter(bias_init(next_key(), (nO,)))
  def __call__(self, x):
    return x @ self.W.load() + self.b.load()

class Sequential(Module):
  def __init__(self, layers):
    self.layers = layers
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

def arch1():
  layer1 = Linear(2, 2)
  layer2 = Linear(2, 2)
  layer1.W = layer2.W
  return Sequential([layer1, layer2])

def arch2():
  layer1 = Linear(2, 2)
  layer2 = layer1
  return Sequential([layer1, layer2])

model1 = inject(arch1, _global_PRNG_key)(random.PRNGKey(1))
print('model1', model1)
model2 = inject(arch2, _global_PRNG_key)(random.PRNGKey(1))
print('model2', model2)

def loss1(x):
  return jnp.sum(model1(x))

def loss2(x):
  return jnp.sum(model2(x))

params1 = tree_load(model1, Parameter)
print('params1', params1)
params2 = tree_load(model2, Parameter)
print('params2', params2)

def train_step1(params, x):
  grads = jax.grad(inject(loss1, model1))(params, x)
  print('grads1', grads)
  return jax.tree_multimap(lambda p, g: p - g, params, grads)

def train_step2(params, x):
  grads = jax.grad(inject(loss2, model2))(params, x)
  print('grads2', grads)
  return jax.tree_multimap(lambda p, g: p - g, params, grads)

x = random.normal(random.PRNGKey(0), (2,))
params1 = train_step1(params1, x)
print('params1', params1)
tree_store(model1, params1)
print('model1', model1)
params2 = train_step2(params2, x)
print('params2', params2)
tree_store(model2, params2)
print('model2', model2)
