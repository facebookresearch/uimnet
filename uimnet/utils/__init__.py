#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import collections
from collections import OrderedDict
import sys
import inspect
from contextlib import AbstractContextManager
import torch.distributed as dist
import unittest
import threading
import torchvision
import subprocess
import submitit
import getpass
import pickle
from pathlib import Path
import shutil
import torch
import types
import shlex
import json
import uuid
import copy
import os
import random
import numpy
import numbers
import functools
import time
import logging
import pandas
# import psutil
from omegaconf import OmegaConf
from torch import is_tensor
from uimnet import __PROJECT_ROOT__
from uimnet import __USE_CHECKPOINTS__
from uimnet import __SLURM_CONFIGS__

# def get_memory_usage():
#     mems = psutil.virtual_memory()
#     unit = 1024**3
#     msg = f'Total/Avail: {mems.total / unit:.2f}/{mems.available / unit:.2f} GB'
#     return msg



def pjson(s, indent=None):
  print(json.dumps(s, indent=indent))


class Identity(torch.nn.Module):
  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class SoftCrossEntropyLoss(torch.nn.Module):
  """
    Cross-Entropy loss accepting soft targets
    """

  def __init__(self):
    super(SoftCrossEntropyLoss, self).__init__()
    self.lsm = torch.nn.LogSoftmax(dim=1)

  def forward(self, logits, targets):
    return (targets * self.lsm(logits)).sum(1).mean().mul(-1)


class Job:
  NOT_LAUNCHED = 'Not launched'
  INCOMPLETE = 'Incomplete'
  DONE = 'Done'

  def __init__(self, callable_, args, dirname):
    self.callable_ = callable_
    self.args = copy.deepcopy(args)

    self.dirname = dirname
    # self.done_file = command[-1].split(".")[-1] + ".done"
    self.done_file = str(callable_) + ".done"

    # for k, v in sorted(self.args.items()):
    #     if isinstance(v, list):
    #         v = ' '.join([str(v_) for v_ in v])
    #     elif isinstance(v, str):
    #         v = shlex.quote(v)
    #     command.append(f'--{k} {v}')
    # self.command_str = ' '.join(command)

    if os.path.exists(os.path.join(self.dirname, self.done_file)):
      self.state = Job.DONE
    elif os.path.exists(self.dirname):
      self.state = Job.INCOMPLETE
    else:
      self.state = Job.NOT_LAUNCHED

  def __str__(self):
    return '{}: {}'.format(self.state, self.dirname)


class JobLauncher:
  def __init__(
      self,
      submitit_dir=None,
      partition="learnfair",
      time=1 * 4 * 60,
      nodes=1,
      ntasks_per_node=1,
      gpus_per_node=1,
      mem=64,
      array_parallelism=512,
      cpus_per_task=8
  ):

    if submitit_dir is None:
      self.submitit_dir = os.path.join(
          "/checkpoint/", getpass.getuser(), "submitit/"
      )
    else:
      self.submitit_dir = submitit_dir

    self.partition = partition
    self.time = time
    self.nodes = nodes

    self.ntasks_per_node = ntasks_per_node
    self.gpus_per_node = gpus_per_node
    self.mem = mem

    self.array_parallelism = array_parallelism
    self.cpus_per_task = cpus_per_task

  def submitit_launcher_(self, jobs, local=False):

    if local:
      raise NotImplementedError
      # for cmd in commands:
      #     subprocess.call(cmd, shell=True)
      # return

    # def _run_command(command):
    #     subprocess.run(command, shell=True)

    sweep_submitit_dir = os.path.join(self.submitit_dir, str(uuid.uuid4()))

    os.makedirs(sweep_submitit_dir, exist_ok=True)

    executor = submitit.SlurmExecutor(folder=sweep_submitit_dir)
    mem_per_gpu = f'{self.mem}G'
    executor.update_parameters(
        time=self.time,
        nodes=self.nodes,
        ntasks_per_node=self.ntasks_per_node,
        gpus_per_node=self.gpus_per_node,
        array_parallelism=self.array_parallelism,
        cpus_per_task=self.cpus_per_task,
        # mem=self.mem,
        mem_per_gpu=mem_per_gpu,
        partition=self.partition
    )
    all_callables, all_args = zip(*[(j.callable_, j.args) for j in jobs])

    executor.map_array(all_callables[0], all_args)

  def ask_for_confirmation_(self):
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
      print('Nevermind!')
      exit(0)

  def print(self, jobs):
    for job in jobs:
      print(job)

    print(
        "{} jobs: {} done, {} incomplete, {} not launched.".format(
            len(jobs), len([j for j in jobs if j.state == Job.DONE]),
            len([j for j in jobs if j.state == Job.INCOMPLETE]),
            len([j for j in jobs if j.state == Job.NOT_LAUNCHED])
        )
    )

  def command(self, command, jobs, local=False, skip_confirmation=False):
    self.print(jobs)

    if command == 'launch':
      self.launch(
          [j for j in jobs if j.state == Job.NOT_LAUNCHED], local,
          skip_confirmation
      )
    elif command == 'relaunch':
      self.launch(
          [j for j in jobs if j.state != Job.DONE], local, skip_confirmation
      )
    elif command == 'relaunch_all':
      self.launch(jobs, local, skip_confirmation)
    elif command == 'delete_incomplete':
      self.delete(
          [j for j in jobs if j.state == Job.INCOMPLETE], skip_confirmation
      )

  def launch(self, jobs, local=False, skip_confirmation=False):
    if not skip_confirmation:
      print(f'About to launch {len(jobs)} jobs.')
      self.ask_for_confirmation_()
    print('Launching...')

    if len(jobs):
      jobs = [jobs[i] for i in torch.randperm(len(jobs))]
      print('Making job directories:')
      for job in jobs:
        os.makedirs(job.dirname, exist_ok=True)

      #commands = [job.command_str for job in jobs]
      self.submitit_launcher_(jobs)

    print(f'Launched {len(jobs)} jobs!')

  def delete(self, jobs, skip_confirmation=False):
    if not skip_confirmation:
      print(f'About to delete {len(jobs)} jobs.')
      self.ask_for_confirmation_()
    print('Deleting...')
    if len(jobs):
      for job in jobs:
        shutil.rmtree(job.dirname)
    print(f'Deleted {len(jobs)} jobs!')


def ensemble_dir(member_dirs):
  member_dirs = [os.path.normpath(d) for d in member_dirs]

  if len(member_dirs) == 1:
    return member_dirs[0]

  dirname = os.path.dirname(member_dirs[0])
  basenames = list(sorted([os.path.basename(d) for d in member_dirs]))
  return os.path.join(dirname, "_".join(basenames))


def is_distributed():
  return dist.is_available() and dist.is_initialized()


class MLP(torch.nn.Module):
  def __init__(self, sizes, batch_norm=False):
    super(MLP, self).__init__()
    layers = []

    for s in range(len(sizes) - 1):
      layers += [
          torch.nn.Linear(sizes[s], sizes[s + 1]),
          torch.nn.BatchNorm1d(sizes[s + 1])
          if batch_norm and s < len(sizes) - 2 else None,
          torch.nn.ReLU()
      ]

    self.network = torch.nn.Sequential(
        *[l for l in layers if l is not None][:-1]
    )

  def forward(self, x):
    return self.network(x)


class NormalizedLinear(torch.nn.Module):
  def __init__(self, in_features, out_features, bias=True):
    super(NormalizedLinear, self).__init__()
    self.layer = torch.nn.Linear(in_features, out_features, bias)

  def forward(self, x):
    x_norms = x.view(len(x), -1).pow(2).sum(1, keepdim=True)
    return self.layer(x).div(x_norms)


class NamespacedDict(dict):
  """
  `dict` with keys as attributes

  """

  def __init__(self, adict: dict):
    super(NamespacedDict, self).__init__(adict.items())
    # Checking if the dict contain any key named as `dict` attribute.
    for key in adict:
      if key in dir(dict):
        msg = f'Dictionary contains reserved key : {key}'
        raise ValueError(msg)

    self.__dict__.update(adict)


class NamespacedOrderedDict(OrderedDict):
  """
  `OrderedDict` Dictionary with keys as attributes

  """

  def __init__(self, adict: OrderedDict):
    super(NamespacedOrderedDict, self).__init__(adict.items())
    # Checking if the dict contain any key named as `dict` attribute.
    for key in adict:
      if key in dir(OrderedDict):
        msg = f'Ordered Dictionary contains reserved key: {key}'
        raise ValueError(msg)

    self.__dict__.update(adict)


def namespace_dict(el: dict):
  """
  Recursively transforms `dict` into a `dict` with keys as attributes.
  Will raise an error if any of the keys is a default attribute of the `dict`
  class.
  """

  if isinstance(el, OrderedDict):
    return NamespacedOrderedDict(
        OrderedDict((k, namespace_dict(v)) for k, v in el.items())
    )

  elif isinstance(el, dict):
    return NamespacedDict({k: namespace_dict(v) for k, v in el.items()})

  else:
    return el


def patch_temperature(algorithm, loader):
  temperature = torch.nn.Parameter(torch.ones(1))
  cross_entropy = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

  all_logits = []
  all_targets = []
  algorithm.eval()
  with torch.no_grad():
    for x, y in loader:
      x = x.to(algorithm.device, non_blocking=True)
      y = y.to(algorithm.device, non_blocking=True)
      all_logits.append(algorithm(x).cpu())
      all_targets.append(y.cpu())
  algorithm.train()
  all_logits = torch.cat(all_logits)
  all_targets = torch.cat(all_targets)

  def closure():
    loss_value = cross_entropy(all_logits / temperature, all_targets)
    loss_value.backward()
    return loss_value

  optimizer.step(closure)

  temperature = temperature.item()

  def new_forward(self, x):
    return self.old_forward(x) / temperature

  algorithm.old_forward = algorithm.forward
  algorithm.forward = types.MethodType(new_forward, algorithm)
  algorithm.temperature = temperature
  return algorithm


class MCReLUDropout(torch.nn.Module):
  def __init__(self, dropout=0.1):
    super(MCReLUDropout, self).__init__()
    self.dropout = dropout

  def forward(self, x):
    return torch.nn.functional.dropout(
        torch.nn.functional.relu(x), self.dropout, training=True
    )


class suppress_stdout(AbstractContextManager):
  """Context manager for suppressing stdout"""
  _stream = "stdout"

  def __init__(self):
    super(suppress_stdout, self).__init__()

    self._devnull = open(os.devnull, 'w')
    self._old_targets = []

  def __enter__(self):
    self._old_targets.append(getattr(sys, self._stream))
    setattr(sys, self._stream, self._devnull)
    return self._devnull

  def __exit__(self, exctype, excinst, exctb):
    self._devnull.close()

    setattr(sys, self._stream, self._old_targets.pop())


def set_random_seed(seed):
  """
  Sets random seed
  """
  random.seed(seed)
  numpy.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


def apply_fun(fun, itr):
  "Apply function to iterable"
  if isinstance(itr, dict):
    return {k: apply_fun(fun, v) for k, v in itr.items()}
  elif isinstance(itr, OrderedDict):
    return OrderedDict([(k, apply_fun(fun, v)) for k, v in list(itr)])
  elif isinstance(itr, list) or isinstance(itr, tuple):
    els = [apply_fun(fun, v) for v in itr]
    return tuple(els) if isinstance(itr, tuple) else els
  elif isinstance(itr, types.GeneratorType):
    return (apply_fun(fun, v) for v in itr)
  else:
    return fun(itr)


def is_numpy(var):
  "Check if variable's type is on of the `numpy` types"
  return type(var).__module__ == numpy.__name__


def is_numpy_array(var):
  "Check if variable's type is `numpy.ndarray` "
  return isinstance(var, numpy.ndarray)


def is_number(var):
  "Check if variable is of type `numbers.Number`"
  return isinstance(var, numbers.Number)


def to_numpy(var):
  "Transform torch.Tensor and torch.autograd.Variable to numpy arrays"
  if is_number(var):
    return var
    # dtype = 'float32' if isinstance(var, float) else None
    # return np.array(var, dtype=dtype)
  elif is_numpy(var):
    return var

  if torch.is_tensor(var):
    return var.detach().cpu().numpy() if var.is_cuda else var.detach().numpy()

  return var

def map_dict(list_of_dicts, op=None):
  """
  Applies operation `op` on the concatenation of the values list of dictionaries sharing the same key.
  """
  ndicts = len(list_of_dicts)
  if ndicts == 0:
    err_msg = f'Expect list of positive length'
    raise ValueError(err_msg)

  if ndicts == 1:
    return list_of_dicts[0]

  out = collections.defaultdict(list)
  for _dict in list_of_dicts:
    for key, val in _dict.items():
      #out[key] += [val]
      out[key] += pack(val)
  out = dict(out)

  if op is None:
    return out

  return {k: op(v) for k, v in out.items()}


def to_scalar(var):
  "Transform torch.Tensor and torch.autograd.Variable to numpy arrays"
  if is_number(var):
    return var
    # dtype = 'float32' if isinstance(var, float) else None
    # return np.array(var, dtype=dtype)
  elif is_numpy(var):
    return float(var)

  if torch.is_tensor(var):
    return float(var.detach().cpu())

  return var

def timeit(fun):
  "Time the function call. Outputs to logger"

  @functools.wraps(fun)
  def wrapped(*args, **kwargs):
    start_time = time.time()
    output = fun(*args, **kwargs)
    elapsed_time = time.strftime('%H:%M:%S',
                                 time.gmtime(time.time() - start_time))

    msg = f'time in {fun.__name__}: {elapsed_time}.'
    message(msg)
    return output

  return wrapped

def pack(arg):
  """
  Pack variables into a list.
  """
  if isinstance(arg, (list, tuple)):
    return list(arg)
  else:
    return [arg]


def make_error_bars_df(means_df, stds_df):
  """
  Outputs a dataframe with latex error bars.
  """
  means = means_df.values.tolist()
  stds = stds_df.values.tolist()
  output_vals = []
  for rows in zip(means, stds):
    output_row = []
    for vals in zip(*rows):
      mean, std = vals
      entry = f'${mean} \\pm {std}$'
      output_row.append(entry)
    output_vals.append(output_row)
  return pandas.DataFrame(data=output_vals, index=means_df.index, columns=means_df.columns)

def make_error_bars_df_with_count(means_df, stds_df, counts_df):
  """
  Outputs a dataframe with latex error bars.
  """
  means = means_df.values.tolist()
  stds = stds_df.values.tolist()
  counts = counts_df.values.tolist()
  output_vals = []
  for rows in zip(means, stds, counts):
    output_row = []
    for vals in zip(*rows):
      mean, std, count = vals
      entry = f'${mean} \pm {std} \, ({count})$'
      output_row.append(entry)
    output_vals.append(output_row)
  return pandas.DataFrame(data=output_vals, index=means_df.index, columns=means_df.columns)

def maybe_synchronize():
  if is_distributed():
    dist.barrier()
  return

def is_not_distributed_or_is_rank0():
  return (not is_distributed() or dist.get_rank() == 0)

def message(string):
  caller = inspect.getframeinfo(inspect.stack()[1][0])

  prefix = f'{caller.filename}@{caller.lineno}: \n'
  if is_distributed():
    prefix = f'(Rank: {dist.get_rank()}, PID: {os.getpid()}) \t'

  msg = f'{prefix}\t{string}'
  print(msg, flush=True)
  return

def to_device(var, device, non_blocking=True):

  if is_tensor(var) or is_module(var):
    return var if is_cuda(var) else var.to(device, non_blocking=non_blocking)
  elif is_numpy(var):
    return torch.from_numpy(to_numpy(var)).to(device, non_blocking=non_blocking)
  elif is_number(var):
    return torch.from_numpy(to_numpy(var)).to(device, non_blocking=non_blocking)

def map_device(var, device, non_blocking=True):
  _to_device = functools.partial(to_device, device=device, non_blocking=non_blocking)
  return apply_fun(_to_device, var)


def is_cuda(var):
  "checks if variable is on gpu"
  if is_tensor(var):
    return var.is_cuda
  elif is_module(var):
    return next(var.parameters()).is_cuda
  else:
    err_msg = 'expected torch.Tensor, torch.cuda.Tensor, or torch.nn.Module got {}'
    raise TypeError(err_msg.format(type(var)))


def is_module(var):
  "Check if variable is of type `torch.nn.Module`"
  return isinstance(var, torch.nn.Module)

def checkpoint(checkpoint_dir=None):
  """
  Serializes function output to file
  """
  checkpoint_dir = os.getenv('CHECKPOINT_DIR', checkpoint_dir)
  if checkpoint_dir is None:
    checkpoint_dir = __PROJECT_ROOT__ + '/checkpoints'

  def _checkpoint(fun):

    @functools.wraps(fun)
    def wrapper(*args, **kwargs):

      filename = f'{fun.__name__}.pkl'
      checkpoint_path = Path(checkpoint_dir)
      checkpoint_path.mkdir(parents=True, exist_ok=True)
      checkpoint_filepath = checkpoint_path / filename
      if checkpoint_filepath.exists() and __USE_CHECKPOINTS__:
        try:
          with open(checkpoint_filepath, 'rb') as fp:
            message(f'Loading checkpoint for {fun.__name__} from {checkpoint_filepath.absolute()}.')
            output = pickle.load(fp)
        except Exception as e:
          err_msg = f'Couldn\'t load {fun.__name__} from {checkpoint_filepath.absolute()}: {e}'
          raise IOError(err_msg)
      else:
        output = fun(*args, **kwargs)
        if __USE_CHECKPOINTS__:
          with open(checkpoint_filepath, 'wb') as fp:
            pickle.dump(output, fp, protocol=pickle.HIGHEST_PROTOCOL)
      return output

    return wrapper

  return _checkpoint

@timeit
def partition_dataset(name:str, root: str, split:str, partitions:dict, equalize_partitions: bool):
  from uimnet import datasets

  tmin = None
  if equalize_partitions:
    tmin = min(list(len(part['targets']) for part in partitions.values()))

  CLS = datasets.__dict__[name]
  dataset_kwargs = dict(root=root, split=split, transform=datasets.TRANSFORMS['eval'])
  dataset = CLS(**dataset_kwargs)

  _datasets = {}
  for partition_name, partition in partitions.items():
    selected_targets = partition['targets'][:tmin]
    partition_indices = dataset.get_targets_indices(selected_targets)
    # TODO(Ishmael): Should we shuffle the indices before subsetting them?
    dataset_partition = CLS(indices=partition_indices, **dataset_kwargs)
    dataset_partition.name = partition_name
    _datasets[dataset_partition.name] = dataset_partition

  return _datasets


def write_trace(trace: str, dir_: str, verbose=True):

  filename_state = trace.split('.')
  if len(filename_state) != 2:
    err_msg = f'trace should be a string of the form filename.value'
    raise ValueError(err_msg)

  path = Path(dir_)
  filepath = path.absolute() /  f'{filename_state[0]}.state'

  with open(filepath, 'w') as fp:
    fp.write(filename_state[1])

  if verbose:
    message(f'Trace serialized @ {filepath}.')
  return filepath


def trace_exists(trace:str, dir_: str):

  filename_state = trace.split('.')
  if len(filename_state) != 2:
    err_msg = f'trace should be a string of the form filename.value'
    raise ValueError(err_msg)

  filepath = Path(dir_).absolute() /  f'{filename_state[0]}.state'

  if not filepath.exists():
    return False

  # File exists
  # load stored value
  with open(filepath, 'r') as fp:
    stored_value = fp.read()

  if stored_value == filename_state[1]:
    return True

  return False



def adapt_state_dict(to_load, to_receive):
  """
  Adapt distributed state dictionaries to non-distributed
  """

  iterator = zip(list(to_load.keys()),
                 list(to_receive.keys()))
  for kl, kr in iterator:
    if kl != kr:
      kl_new = kl.replace('.module', '')
      assert kl_new == kr
      to_load[kl_new] = to_load.pop(kl)
  return to_load


def flatten_nested_dicts(d, parent_key='', sep='.'):
  items = []
  for k, v in d.items():
      new_key = parent_key + sep + k if parent_key else k
      if isinstance(v, collections.MutableMapping):
          items.extend(flatten_nested_dicts(v, new_key, sep=sep).items())
      else:
          items.append((new_key, v))
  return dict(items)


def is_valid_subpath(subpath):

  if not subpath.exists():
    return False

  if trace_exists('train.done', dir_=str(subpath)):
    return True

  return False


def scrap(obj):
  _lambda = lambda el: not any([el.startswith(e) for e in _exclude])
  for attr in filter(_lambda, dir(obj)):
    delattr(obj, attr)
  return

def load_cfg(path, sweep_dir=None):
  with open(path, 'r') as fp:
    cfg = OmegaConf.load(fp.name)

  OmegaConf.set_readonly(cfg, False)  # Write protect.
  OmegaConf.set_struct(cfg, True)  # Raise missing keys error.

  if sweep_dir is not None:
    if hasattr(cfg, 'sweep_dir') and cfg.sweep_dir is not None:
      message(f'Overriding config file sweep dir {cfg.sweep_dir} with {sweep_dir}.')
    cfg.sweep_dir = sweep_dir

  OmegaConf.set_readonly(cfg, True)  # Write protect.
  OmegaConf.set_struct(cfg, True)  # Raise missing keys error.
  return cfg

def load_cfgs(cfgs_dir, sweep_dir=None):

  configs_path = Path(cfgs_dir)
  cfgs_paths  = {el.stem.split('_')[1]: el.absolute() for el in configs_path.iterdir()}
  return {k: load_cfg(p, sweep_dir) for k, p in cfgs_paths.items()}

def load_model_cls(train_cfg):
  from uimnet import ensembles
  from uimnet import algorithms

  model_path = Path(train_cfg.output_dir)
  # TODO(Ishmael): Concentrate ensembles checking
  if 'ensemble' in train_cfg and train_cfg.ensemble.name == 'Bagging':
    Ensemble = ensembles.Bagging
    with open(model_path / 'paths.pkl', 'rb') as fp:
      paths = pickle.load(fp)
    return functools.partial(Ensemble, paths=paths)
  else:
    return algorithms.__dict__[train_cfg.algorithm.name]


def parametrize(Test, parameters):
  parameters = pack(parameters)

  def _parametrize(Test, parameters: dict):
    cls_name = f'{Test.__name__[1:]}' + '_' +'_'.join([f'{k}_{str(v)}' for k, v in parameters.items()])
    return type(cls_name, (Test, unittest.TestCase), parameters)

  return [_parametrize(Test, params) for params in parameters]


def report(job, path, stem):

  with open(path / f'{stem}.out', 'w') as fp:
    msg = job.stdout()
    fp.write(msg if msg is not None else '')

  with open(path / f'{stem}.err', 'w') as fp:
    msg = job.stderr()
    fp.write(msg if msg is not None else '')

  return


class Beholder(threading.Thread):
  def __init__(self, job_paths, stem):
    super(Beholder, self).__init__()
    self.job_paths = job_paths
    self.stem = stem

  def run(self):
    job_paths = self.job_paths
    while len(job_paths) > 0:
      for i, (job, path) in enumerate(job_paths):
        try:
          report(job, path, stem=self.stem)
          if job.done():
            report(job, path, stem=self.stem)
            job_paths.pop(i)
        except Exception as e:
          message(e)
        time.sleep(1.)


class ExtractRecords(object):
  def __call__(self, records_filepath):
    load_fun = {'.pth': torch.load,
                '.pkl': pickle.load}[records_filepath.suffix]
    with open(records_filepath, 'rb') as fp:
      record = load_fun(fp)
      return apply_fun(to_scalar, record)

def train_done(path):
  if trace_exists('train.done', dir_=str(path)):
    return True
  return False

def prediction_done(path):
  if trace_exists('prediction.done', dir_=str(path)):
    return True
  return False

def calibration_done(path):
  if trace_exists('calibration.done', dir_=str(path)):
    return True
  return False

def is_algorithm(path):
  if (path / '.algorithm').exists():
    return True
  return False

def is_ensemble(path):
  if (path / '.ensemble').exists():
    return True
  return False

def is_model(path):
  if is_algorithm(path) or is_ensemble(path):
    return True
  return False

def identity(x):
  """Identity function"""
  return x

def compose(*funcs):
  """ Function composition
  """
  if len(funcs) == 0:
    raise ValueError('No functions to compose')

  def _compose(g, f):
    return lambda x: g(f(x))

  return functools.reduce(_compose, funcs, identity)


def get_slurm_executor(slurm_config, log_folder):
  slurm_preset = slurm_config.pop('preset')
  executor = submitit.SlurmExecutor(folder=log_folder)
  executor.update_parameters(
    **__SLURM_CONFIGS__[slurm_preset]
  )
  comment = slurm_config.pop('comment', 'Yippee ki-yay')
  executor.update_parameters(
    comment=comment,
    **slurm_config
  )
  return executor


def handle_jobs(jobs):
  total_jobs = len(jobs)
  finished_jobs = []
  while len(jobs) > 0:
    for i, job in enumerate(jobs):
      try:
        if job.done():
          finished_jobs.append(jobs.pop(i))
          message(f'Finished {len(finished_jobs)} / {total_jobs}. {len(jobs)} remaining.')
      except submitit.core.utils.FailedSubmissionError as e:
        # TODO(Ishmael): Resubmit failed submissions
        message(e)
      except submitit.core.utils.FailedJobError as e:
        message(e)
      except Exception as e:
        message(e)
  return finished_jobs, jobs


def all_gather(tensor):

  ranks = list(range(dist.get_world_size()))

  N = torch.tensor([tensor.size(0)]).long().cuda()
  all_N = [torch.zeros_like(N) for _ in ranks]
  dist.all_gather(all_N, tensor=N)
  max_N = max([el.item() for el in all_N])

  padded_shape = (max_N, ) + tensor.shape[1:]
  padded_tensors = [torch.zeros(padded_shape, device=tensor.device, dtype=tensor.dtype) for _ in ranks]

  dist.all_gather(padded_tensors, tensor)

  return [t[:s] for t, s in zip(padded_tensors, all_N)]


def all_close(tensor):


  if not is_distributed() or dist.get_world_size() == 1:
    return True

  tensor = tensor.contiguous()
  tensors = all_gather(tensor)
  return all([torch.allclose(t, tt) for t, tt in zip(tensors[:-1], tensors[1:])])


def all_cat(tensors, dim):
  """
  Concatenates locally and across workers
  """
  local_cat = torch.cat(tensors, dim=dim)

  if not is_distributed() or dist.get_world_size() == 1:
    return local_cat

  all_cats = all_gather(local_cat)
  cat = torch.cat(all_cats, dim=dim)

  if not all_close(cat):
    err_msg = f'Unequal tensors across workers'
    raise RuntimeError(err_msg)

  return cat

def guard(config):
  OmegaConf.set_struct(config, True)
  OmegaConf.set_readonly(config, True)
  return config

if __name__ == '__main__':
  pass
