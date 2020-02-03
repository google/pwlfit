# Lint as: python2, python3
# Copyright 2019 Google LLC
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


"""Routines for picking transform functions over a weighted data set."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pwlfit import utils


def weighted_pearson_correlation(x, y, w):
  """Computes weighted correlation between x and y.

  Args:
    x: (numerical numpy array) First feature for the correlation.
    y: (numerical numpy array) Second feature for the correlation.
    w: (numerical numpy array) Weights of the points.

  Raises:
    ValueError: if x, y, w have different lengths or are empty.

  Returns:
    The weighted correlation between x and y, from -1 (perfect inverse
    correlation) to 1 (perfect correlation). 0 indicates no correlation.
    NaN if all x-values or all y-values are the same.
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
  """
  utils.expect(len(x) == len(y) == len(w) >= 1,
               'x, y, and w must be nonempty and equal in length.')
  x, y, w = np.asarray(x), np.asarray(y), np.asarray(w)
  xm = x - np.average(x, weights=w)
  ym = y - np.average(y, weights=w)
  if (xm == xm[0]).all() or (ym == ym[0]).all():
    # Correlation isn't defined when one variable is constant.
    return float('nan')

  covxy = np.average(xm * ym, weights=w)
  covxx = np.average(xm ** 2, weights=w)
  covyy = np.average(ym ** 2, weights=w)
  return covxy / np.sqrt(covxx * covyy)


def symmetriclog1p(x):
  """An extension of log1p(x >= 0) to all real x.

  Args:
    x: (Numpy array of floats) Data to transform.

  Returns:
    Numpy array of the symmetriclog1p of each entry in x.
  """
  return np.sign(x) * np.log1p(np.abs(x))


def identity(x):
  return x


def _clip_weight_extremes(ws, pct_to_clip):
  """Clips the pct_to_clip first and last values by weight."""
  utils.expect(0 <= pct_to_clip < .5)
  if pct_to_clip == 0:
    return ws

  ws = np.array(ws, copy=True, dtype=float)
  ws_cumsum = ws.cumsum()
  w_sum = ws_cumsum[-1]
  cut_weight = w_sum * pct_to_clip

  # The indices of the first and last nonzero entries after clipping.
  first_nonzero = ws_cumsum.searchsorted(cut_weight, side='right')
  last_nonzero = ws_cumsum.searchsorted(w_sum - cut_weight, side='left')
  ws[:first_nonzero], ws[last_nonzero + 1:] = 0, 0

  # Use any leftover cut_weight to reduce the weights at first and last nonzero.
  ws[first_nonzero] = ws_cumsum[first_nonzero] - cut_weight
  ws[last_nonzero] = w_sum - ws_cumsum[last_nonzero - 1] - cut_weight
  return ws, first_nonzero, last_nonzero


def find_best_transform(sorted_x, y, w, pct_to_clip=0.005, identity_bias=1e-6):
  """Finds the best transformation to use for PWLCurve fitting.

  Chooses between identity, log, logp1, and symmetriclogp1.

  Args:
    sorted_x: (Numpy array) Data to transform, in sorted order.
    y: (Numpy array) Dependent variable of data to transform.
    w: (Numpy array) Weights of data to transform.
    pct_to_clip: (float) The fraction of upper and lower outliers to ignore.
        Necessary because the Pearson correlation is susceptible to x-outliers.
    identity_bias: (float) The amount by which we prefer the identity. By
        default, this is an epsilon used to break effective ties.

  Returns:
    Chosen transform_fn to apply to sorted_x.
  """
  if len(sorted_x) <= 1:
    return identity

  clipped_w, first_nonzero, last_nonzero = _clip_weight_extremes(w, pct_to_clip)
  if sorted_x[first_nonzero] == sorted_x[last_nonzero]:
    # No reason to transform if there's only one unique x after clipping.
    return identity

  # Pick the domain-appropriate log transform.
  if sorted_x[0] > 0:
    transform = np.log
  elif sorted_x[0] > -1:
    transform = np.log1p
  else:
    transform = symmetriclog1p

  # Use the Pearson correlation to find the transformation on which the data
  # looks the most linear.
  # https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
  identity_pearson = weighted_pearson_correlation(sorted_x, y, clipped_w)
  log_pearson = weighted_pearson_correlation(transform(sorted_x), y, clipped_w)
  if abs(log_pearson) > abs(identity_pearson) + identity_bias:
    return transform

  return identity
