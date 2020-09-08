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


"""Utilities for condensing arbitrarily many points to two.

Any set of points P can be compressed to a pair of points P' such that fitting a
line on P' is equivalent to fitting the line on P. More specifically, for any
two lines L1 and L2:
  MSE(L1 over P) - MSE(L2 over P) = MSE(L1 over P') - MSE(L2 over P').

P' has the same centroid, total weight, and best fit line as P. Additionally,
we choose P' such that min_x(P) <= min_x(P') <= max_x(P') <= max_x(P). This
property allows us to extend linear condensing from line fitting to PWLCurves.

"""

import math
from typing import Sequence, Tuple

import numpy as np
from pwlfit import utils


def _recenter_at_zero(
    x: np.ndarray, y: np.ndarray,
    w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
  """Shifts the centroid to zero.

  Args:
    x: (numpy array) X-values to center at zero.
    y: (numpy array) Y-values to center at zero.
    w: (numpy array) the weights on data points.

  Returns:
    A tuple of two numpy arrays and two floats (x - cx, y - cy, cx, cy), where
    (cx, cy) is the centroid of (x, y, w).
  """
  w_sum = w.sum()
  cx, cy = np.dot(x, w) / w_sum, np.dot(y, w) / w_sum
  return x - cx, y - cy, cx, cy


def linear_condense(
    x: np.ndarray, y: np.ndarray,
    w: np.ndarray) -> Tuple[Sequence[float], Sequence[float], Sequence[float]]:
  """Returns X', Y', W' that replicates the linear fit MSE of x, y, w.

  This function compresses an arbitrary number of (x,y,weight) points into at
  most two compressed points, such that the difference of the MSEs of any two
  lines is the same on the compressed points as it is on the given points. The
  two compressed points lie on the best fit line of the original points, and
  have the same centroid and weight sum as the original points. They also meet
  other invariants derived through linear algebra.

  Args:
    x: (numpy array) independent variable.
    y: (numpy array) dependent variable.
    w: (numpy array) the weights on data points. Each weight must be positive.

  Returns:
    A tuple of 3 lists X', Y', W', each of length no greater than two, such that
    the different of the MSEs of any two lines is the same on X', Y', W' as it
    is on x, y, w.

  """
  utils.expect(len(x) == len(y) == len(w))
  x = np.array(x, dtype=float, copy=False)
  y = np.array(y, dtype=float, copy=False)
  w = np.array(w, dtype=float, copy=False)
  if len(x) <= 2:
    return x, y, w

  # Math is simpler and stabler if we recenter x and y to a zero centroid.
  # We'll add centroid_x and centroid_y back at the end.
  x, y, centroid_x, centroid_y = _recenter_at_zero(x, y, w)
  weight_sum = w.sum()
  centroid_as_fallback = ([centroid_x], [centroid_y], [weight_sum])

  xmin, xmax = np.min(x), np.max(x)
  if not xmin < 0 < xmax:
    return centroid_as_fallback

  xxw = np.dot(x * x, w)
  x_variance = xxw / weight_sum  # Weighted variance.

  x1 = -math.sqrt(x_variance * -xmin / xmax)
  x2 = math.sqrt(x_variance * xmax / -xmin)

  w1 = weight_sum * xmax / (xmax - xmin)
  w2 = weight_sum - w1
  if w1 <= 0 or w2 <= 0:  # Only possible if float precision fails.
    return centroid_as_fallback

  # The best fit line passes through the centroid, so the y-intercept is 0.
  bestfit_slope = np.dot(x * y, w) / xxw
  y1 = bestfit_slope * x1
  y2 = bestfit_slope * x2

  # Add back the centroid of the original data.
  return ([x1 + centroid_x, x2 + centroid_x],
          [y1 + centroid_y, y2 + centroid_y],
          [w1, w2])


def _condense_between_indices(
    sorted_x: np.ndarray, y: np.ndarray, w: np.ndarray,
    sorted_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Condenses the points between each pair of indices into at most two."""
  condensed_x, condensed_y, condensed_w = [], [], []  # Output aggregators.

  for start, end in zip(sorted_indices[:-1], sorted_indices[1:]):
    region_xs, region_ys, region_ws = linear_condense(
        sorted_x[start:end], y[start:end], w[start:end])
    condensed_x.extend(region_xs)
    condensed_y.extend(region_ys)
    condensed_w.extend(region_ws)

  return np.array(condensed_x), np.array(condensed_y), np.array(condensed_w)


def condense_around_knots(
    sorted_x: np.ndarray, y: np.ndarray, w: np.ndarray,
    sorted_knots: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns X', Y', W' that replicates the PWLFit MSE of x, y, w, knots.

  This function compresses an arbitrary number of (x,y,weight) points into at
  most 2 * (len(sorted_knots) - 1) compressed points, such that the difference
  of the MSEs of any two PWLCurves defined on those knots is the same on the
  compressed points as it is on the given points.
    O(len(sorted_x)).

  Args:
    sorted_x: (numpy array) independent variable in sorted order.
    y: (numpy array) dependent variable.
    w: (numpy array) the weights on data points. Each weight must be positive.
    sorted_knots: (numpy array) x-values of the candidate knots in sorted order.

  Returns:
    A tuple of 3 lists x', y', w', each of equal length no greater than
    2 * (len(sorted_knots) - 1), such that, given any two PWLCurves c1 and c2
    with x-knots from sorted_knots, the difference of the MSEs of c1 and c2
    is the same when evaluated on x', y', w' as it is on x, y, w.

  """
  if sorted_x[0] < sorted_knots[0] or sorted_knots[-1] < sorted_x[-1]:
    # Clamp sorted_x to the range of knot xs before condensing.
    sorted_x = np.clip(sorted_x, sorted_knots[0], sorted_knots[-1])

  sorted_x, y, w = utils.fuse_sorted_points(sorted_x, y, w)
  knot_indices = sorted_x.searchsorted(sorted_knots)

  knot_indices[-1] = len(sorted_x)
  return _condense_between_indices(sorted_x, y, w, knot_indices)


def _greedy_weight_percentile_indices(w: np.ndarray,
                                      target_num_uniques: int) -> np.ndarray:
  """Returns indices for weight percentiles with up to target_num_uniques.

  Chooses indices spaced equally by weight percentiles, including the first and
  and last index. If the selection contains fewer than target_num_uniques unique
  indices, _greedy_weight_percentile_indices iteratively resamples with a higher
  sampling rate until it samples too many unique indices. Each iteration
  includes all the sampled indices of the previous iteration.

  Args:
    w: (numpy array) Weights of the indices to sample in sorted order.
    target_num_uniques: (int) Maximum number of unique samples to return.

  Returns:
    An array of indices in sorted order, representing a sample stratified by w,
    with at most target_num_uniques unique values.
  """
  # First, check if we need to sample at all.
  if len(w) <= target_num_uniques:
    return np.arange(len(w))

  # Iteratively increase number of samples until we get enough uniques.
  num_samples = target_num_uniques
  w_cumsum = w.cumsum()
  while True:
    weight_percentiles = np.linspace(0, w_cumsum[-1], num_samples)
    indices = w_cumsum.searchsorted(weight_percentiles)
    num_uniques = utils.count_uniques_on_sorted(indices)

    if num_uniques > target_num_uniques:
      return prev_indices  # pylint: disable=used-before-assignment
    if num_uniques == target_num_uniques:
      return indices

    prev_indices = indices
    num_samples = num_samples * 2 - 1


def _pick_knot_candidates(x: np.ndarray, w: np.ndarray,
                          num_candidates: int) -> np.ndarray:
  """Pick knots equally spaced by weight percentile."""
  knot_indices = _greedy_weight_percentile_indices(w, num_candidates)
  knot_indices = utils.unique_on_sorted(knot_indices)  # Only keep uniques.
  return x[knot_indices]


def sample_condense_points(
    sorted_x: np.ndarray, y: np.ndarray, w: np.ndarray,
    num_knots: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Picks knots and linearly condenses (sorted_x, y, w) around those knots.

  Args:
    sorted_x: (numpy array) independent variable in sorted order.
    y: (numpy array) dependent variable.
    w: (numpy array) the weights on data points.
    num_knots: (int) Number of knot-x candidates to return.

  Returns:
    A tuple of 4 lists: x_knots, condensed_x, condensed_y, condensed_w.
  """
  utils.expect(num_knots >= 2, 'num_knots must be at least 2.')
  utils.expect(len(sorted_x) == len(y) == len(w))

  sorted_x, y, w = utils.fuse_sorted_points(sorted_x, y, w)
  if len(sorted_x) <= num_knots:
    return sorted_x, sorted_x, y, w

  knot_xs = _pick_knot_candidates(sorted_x, w, num_knots)
  condensed_x, condensed_y, condensed_w = (
      condense_around_knots(sorted_x, y, w, knot_xs))
  return knot_xs, condensed_x, condensed_y, condensed_w
