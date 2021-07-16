# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper utilities to be used in pwlfit-related notebooks."""
import abc
import collections
import re
import timeit
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd
from pwlfit import pwlcurve
from pwlfit import transform
from pwlfit import utils


def _check_has_column(column_name: str, df: pd.DataFrame):
  if column_name not in df.columns:
    raise ValueError('Column [%s] not found in DataFrame!' % column_name)


def _maybe_get_column(column_name: str, data: Union[pd.DataFrame,
                                                    np.ndarray]) -> np.ndarray:
  if isinstance(data, pd.DataFrame):
    _check_has_column(column_name, data)
    return data[column_name].to_numpy()
  return data


class Model(abc.ABC):
  """Abstract base class for models."""

  @abc.abstractmethod
  def eval(self, data) -> np.ndarray:
    raise NotImplementedError()

  def predict(self, data) -> np.ndarray:
    return self.eval(data)

  @abc.abstractproperty
  def name(self) -> str:
    raise NotImplementedError()

  @property
  def clean_name(self) -> str:
    return re.sub('[^0-9a-zA-Z]', '_', self.name)

  @abc.abstractmethod
  def expr(self) -> str:
    raise NotImplementedError()

  def __str__(self) -> str:
    return self.name

  def __eq__(self, o):
    raise NotImplementedError()


class EnumCurveModel(Model):
  """An immutable model representing a discrete mapping."""

  def __init__(self, feature_name: str, control_points: Mapping[int, float]):
    self._feature_name = feature_name
    self._mapping = dict(control_points)
    super(EnumCurveModel, self).__init__()

  def __eq__(self, o):
    return (isinstance(o, EnumCurveModel) and
            (o._feature_name == self._feature_name) and
            (o._mapping == self._mapping))

  @property
  def feature_name(self) -> str:
    return self._feature_name

  @property
  def name(self) -> str:
    return 'f_%s' % self._feature_name

  def eval(self, data: Union[pd.DataFrame, Sequence[float]]) -> np.ndarray:
    return np.fromiter(
        (self._mapping[x] for x in _maybe_get_column(self._feature_name, data)),
        dtype=np.float,
        count=len(data))

  def expr(self) -> str:
    entries = ', '.join([
        '%r: %r' % (k, self._mapping[k]) for k in sorted(self._mapping.keys())
    ])
    return 'EnumCurve("%s", {%s})' % (self._feature_name, entries)

  @classmethod
  def from_expr(cls, expr: str) -> 'EnumCurveModel':
    """Parse model from expression string."""
    my_locals = {}
    exec('res = ' + expr, {'EnumCurve': EnumCurveModel}, my_locals)  # pylint: disable=exec-used for illustration only
    return my_locals['res']


class PWLCurveModel(Model):
  """An immutable model wrapping a PWLCurve."""

  def __init__(self,
               feature_name: str,
               control_points: Sequence[Tuple[float, float]],
               fx: Union[Callable[[Sequence[float]], Sequence[float]],
                         str] = transform.identity):
    self._feature_name = feature_name
    if isinstance(fx, str):
      fx = pwlcurve.PWLCurve.STR_TO_FX[fx]
    self._curve = pwlcurve.PWLCurve(control_points, fx)
    super(PWLCurveModel, self).__init__()

  def __eq__(self, o):
    return (isinstance(o, PWLCurveModel) and
            (o._feature_name == self._feature_name) and
            (o.curve == self._curve))

  @property
  def feature_name(self) -> str:
    return self._feature_name

  @property
  def name(self) -> str:
    return 'f_%s' % self._feature_name

  @property
  def curve(self) -> pwlcurve.PWLCurve:
    return self._curve

  def eval(self, data: Union[pd.DataFrame, Sequence[float]]) -> np.ndarray:
    return self._curve.eval(_maybe_get_column(self._feature_name, data))

  def expr(self):
    res = str(self._curve)
    return res.replace('PWLCurve(', 'PWLCurve("%s", ' % self._feature_name)

  @classmethod
  def from_expr(cls, expr: str) -> 'PWLCurveModel':
    """Parse model from expression string."""
    my_locals = {}
    exec('res = ' + expr, {'PWLCurve': PWLCurveModel}, my_locals)  # pylint: disable=exec-used for illustration only
    return my_locals['res']


class ColumnModel(Model):
  """An immutable model wrapping a column from a DataFrame."""

  def __init__(self, feature_name: str):
    self._feature_name = feature_name
    super(ColumnModel, self).__init__()

  def __eq__(self, o):
    return isinstance(o, ColumnModel) and o._feature_name == self._feature_name

  @property
  def feature_name(self) -> str:
    return self._feature_name

  @property
  def name(self) -> str:
    return self._feature_name

  def eval(self, data: pd.DataFrame) -> np.ndarray:
    _check_has_column(self._feature_name, data)
    return data[self._feature_name].to_numpy()

  def expr(self):
    return 'Column("%s")' % self._feature_name

  @classmethod
  def from_expr(cls, expr: str) -> 'ColumnModel':
    """Parse model from expression string."""
    my_locals = {}
    exec('res = ' + expr, {'Column': ColumnModel}, my_locals)  # pylint: disable=exec-used for illustration only
    return my_locals['res']


class AdditiveModel(Model):
  """An immutable model which is a sum of model with a most one per feature."""

  def __init__(self, submodels: List[Union[ColumnModel, PWLCurveModel,
                                           EnumCurveModel]], name: str):
    self._name = name
    utils.expect(
        len(set(model.feature_name for model in submodels)) == len(submodels),
        'Duplicate submodels for some features')
    self._submodels = collections.OrderedDict([
        (model.feature_name, model)
        for model in sorted(submodels, key=lambda m: m.feature_name)
    ])
    super(AdditiveModel, self).__init__()

  def __eq__(self, o):
    return (isinstance(o, AdditiveModel) and o._name == self._name and
            self._submodels == o._submodels)

  def eval(self, data: pd.DataFrame) -> np.array:
    return sum(model.eval(data) for model in self._submodels.values())

  @property
  def name(self) -> str:
    return self._name

  @property
  def feature_names(self) -> List[str]:
    return sorted(list(self._submodels.keys()))

  def get_submodel(self, feature_name: str) -> Model:
    return self._submodels[feature_name]

  def expr(self) -> str:
    submodel_statements = '\n'.join(
        '  %s,' % model.expr() for model in self._submodels.values())
    return 'score = sum([\n%s\n])' % submodel_statements

  @classmethod
  def from_expr(cls, expr: str, name: str) -> 'AdditiveModel':
    """Parse model from expression string."""
    my_locals = {}
    exec(  # pylint: disable=exec-used for illustration only
        expr, {
            'sum': lambda x: x,
            'PWLCurve': PWLCurveModel,
            'EnumCurve': EnumCurveModel,
            'Column': ColumnModel
        }, my_locals)
    return cls(my_locals['score'], name)


def approx_sample(num_items: int, num_samples: int) -> np.array:
  """Fast approximate downsampling."""
  if num_items <= num_samples:
    return np.ones(num_items, dtype=np.bool8)
  np.random.seed(125)
  # Select each xy with probability (downsample_to / len(x)) to yield
  # approximately downsample_to selections.
  fraction_kept = float(num_samples) / num_items
  return np.random.sample(size=num_items) < fraction_kept


def _cdf(vals):
  vals = np.array(vals)
  return np.cumsum(np.ones_like(vals)) / len(vals)


def _plot_pwlcurve_fit(ax,
                       df: pd.DataFrame,
                       feature_name: str,
                       original_model: Model,
                       curve_model: PWLCurveModel,
                       num_samples: int = 1000):
  """Plot fits for a single PWLCurve."""
  mask = approx_sample(len(df[feature_name]), num_samples=num_samples)

  xs = df[feature_name][mask]
  sorted_xs = np.sort(xs)
  ax.plot(xs, original_model.eval(df.loc[mask]), 'o', alpha=0.5)
  ax.plot(sorted_xs, curve_model.eval(sorted_xs), '-')
  ax.set_xlabel(feature_name)
  ax2 = ax.twinx()
  ax2.grid(False)
  ax2.plot(sorted_xs, _cdf(sorted_xs), '-', color='gray')
  ax2.set_ybound(-0.05, 1.05)
  if curve_model.curve.fx != transform.identity:
    ax.set_xscale('log' if sorted_xs.min() > 0 else 'symlog')
    ax.set_xbound(sorted_xs.min(), sorted_xs.max())
    ax.margins(x=0.05)
    formatter = ticker.ScalarFormatter()
    formatter.set_powerlimits((-3, 4))
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(ticker.NullLocator())

  plt.close(ax.figure)
  return ax, ax2


def _plot_enumcurve_fit(ax, df: pd.DataFrame, feature_name: str,
                        original_model: Model, curve_model: EnumCurveModel):
  """Plot fits for a single EnumCurve."""
  colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
  unique_xs, unique_indicies = np.unique(df[feature_name], return_index=True)
  sorted_indicies = np.argsort(unique_xs)
  sorted_xs = unique_xs[sorted_indicies]
  freq = np.fromiter(
      (np.count_nonzero(df[feature_name] == x) / len(df) for x in sorted_xs),
      float, len(sorted_xs))
  width = 0.5
  ax.scatter(
      sorted_indicies,
      original_model.eval(df.iloc[unique_indicies, :])[sorted_indicies],
      marker='o',
      color=colors[0])
  ax.hlines(
      curve_model.eval(sorted_xs),
      sorted_indicies - width,
      sorted_indicies + width,
      color=colors[1])
  ax.set_xticks(sorted_indicies)
  ax.set_xlabel(feature_name)
  ax2 = ax.twinx()
  ax2.grid(False)
  cdf = np.cumsum(freq)
  ax2.vlines(
      sorted_indicies,
      np.concatenate(([0], cdf[:-1])),
      cdf,
      color='gray',
      linestyles='dashed',
      zorder=1)
  ax2.hlines(cdf[:-1], sorted_indicies[:-1], sorted_indicies[1:],
             color='gray', zorder=1)
  ax2.scatter(
      sorted_indicies[1:],
      cdf[:-1],
      marker='o',
      color='white',
      edgecolor='gray',
      zorder=2)
  ax2.scatter(sorted_indicies, cdf, marker='o', color='gray', zorder=2)
  ax2.set_ybound(-0.05, 1.05)
  plt.close(ax.figure)
  return ax, ax2


def plot_fits(df: pd.DataFrame,
              model_pairs_to_plot: List[Tuple[Model, Union[PWLCurveModel,
                                                           EnumCurveModel]]]):
  """Plots fits vs original model."""
  num_pairs = len(model_pairs_to_plot)
  num_cols = min(3, num_pairs)
  num_rows = (num_pairs + num_cols - 1) // num_cols

  fig, axes = plt.subplots(
      nrows=num_rows, ncols=num_cols, figsize=(5 * num_cols, 5 * num_rows))

  if num_rows == 1:
    axes = [axes]
  idx = 0
  for row in axes:
    for j, ax in enumerate(row):
      if idx >= num_pairs:
        fig.delaxes(ax)
        continue
      original_model, fit_model = model_pairs_to_plot[idx]
      if isinstance(fit_model, PWLCurveModel):
        ax, ax2 = _plot_pwlcurve_fit(ax, df, fit_model.feature_name,
                                     original_model, fit_model)
      else:
        ax, ax2 = _plot_enumcurve_fit(ax, df, fit_model.feature_name,
                                      original_model, fit_model)
      if j == 0:
        ax.set_ylabel('Contribution')
      elif j == num_cols - 1 or idx == num_pairs - 1:
        ax2.set_ylabel('CDF')
      idx += 1
  fig.tight_layout()
  plt.close(fig)
  return fig


def _fit_mapping(xs: Sequence[int], ys: Sequence[float]) -> Dict[int, float]:
  """Fits each x to mean y."""
  xs, ys = np.array(xs), np.array(ys)
  keys = np.unique(xs)
  values = np.round(
      np.fromiter((np.mean(ys[xs == k]) for k in keys),
                  dtype=np.float,
                  count=len(keys)), 4)
  return dict(zip(keys, values))


def fit_additive_model(
    df: pd.DataFrame,
    continuous_targets: Mapping[str, str],
    discrete_targets: Mapping[str, str],
    curve_fitter: Callable[[Sequence[float], Sequence[float]],
                           pwlcurve.PWLCurve],
    name: str,
) -> Tuple[AdditiveModel, Dict[str, float]]:
  """Creates an AdditiveModel composed of EnumCurveModels and PWLCurveModels."""
  submodels = list()
  times = dict()
  for x_name, y_name in continuous_targets.items():
    start = timeit.default_timer()
    curve = curve_fitter(df[x_name], df[y_name])
    elapsed = timeit.default_timer() - start
    submodels.append(PWLCurveModel(x_name, curve.points, curve.fx))
    times[submodels[-1].name] = elapsed
  for x_name, y_name in discrete_targets.items():
    start = timeit.default_timer()
    mapping = _fit_mapping(df[x_name], df[y_name])
    elapsed = timeit.default_timer() - start
    submodels.append(EnumCurveModel(x_name, mapping))
    times[submodels[-1].name] = elapsed
  return AdditiveModel(submodels, name), times


def _dcg_metric(y_true: np.ndarray, y_score: np.ndarray,
                y_size: np.ndarray) -> np.ndarray:
  """Computes discounted cumulative gain."""
  discount = 1 / np.log2(np.arange(y_true.shape[1]) + 2)
  ranking = np.argsort(y_score)[:, ::-1]
  gains = y_true[np.arange(ranking.shape[0])[:, np.newaxis], ranking]
  if np.all(y_size == y_size[0]):
    gains[:, y_size[0]:] = 0
  else:
    for i in range(y_true.shape[0]):
      gains[i, y_size[i]:] = 0
  cumulative_gains = (np.exp2(gains) - 1).dot(discount)
  return cumulative_gains


def ndcg_metric(y_true: np.ndarray,
                y_score: np.ndarray,
                k: int,
                y_size: Optional[np.ndarray] = None) -> np.ndarray:
  """Computes normalized discounted cumulative gain (NDCG).

  https://en.wikipedia.org/wiki/Discounted_cumulative_gain

  Args:
    y_true: Labels. Each row is one query. For varied query length use -inf to
      pad and the y_size option.
    y_score: Scores. Each row is one query. For varied query length use -inf to
      pad and the y_size option.
    k: The depth to compute NDCG to.
    y_size: Optional specification of per query size. For rows i where y_size[i]
      < y_score.shape[1] the padding values in y_true and y_score must be -inf.

  Returns:
    NDCG for each row.
  """
  if y_size is not None:
    y_size = np.minimum(y_size, k)
  else:
    y_size = np.full(y_true.shape[0], k)
  return (_dcg_metric(y_true, y_score, y_size) /
          _dcg_metric(y_true, y_true, y_size))
