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


"""Routines for approximating data with piecewise linear curves."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import isotonic
import linear_condense
import transform
import utils
import scipy.optimize
from six.moves import map
from six.moves import range
from six.moves import zip


def fit_pwl(x,
            y,
            w=None,
            num_segments=3,
            num_samples=100,
            mono=True,
            min_slope=None,
            max_slope=None,
            x_transform=None):
  """Fits a PWLCurve from x to y, minimizing weighted MSE.

  Attempts to find a piecewise linear curve which is as close to ys as possible,
  in a least squares sense.

  ~O(len(x) + qlog(q) + (num_samples^2)(num_segments^3)) time complexity, where
  q is q is ~min(10**6, len(x)). The len(x) term occurs because of downsampling
  to q points. The qlog(q) term comes from sorting after downsampling. The other
  term comes from fit_pwl_points, which greedily searches for the best
  combination of knots and solves a constrained linear least squares expression
  for each.

  Args:
    x: (numpy array) independent variable.
    y: (numpy array) dependent variable.
    w: (numpy array) the weights on data points.
    num_segments: (positive int) Number of linear segments. More segments
      increases quality at the cost of complexity.
    num_samples: (positive int) Number of potential knot locations to try for
      the PWL curve. More samples improves fit quality, but slows fitting. At
      100 samples, fit_pwl runs in 1-2 seconds. At 1000 samples, it runs in
      under a minute. At 10,000 samples, expect an hour.
    mono: (boolean) Whether to require a monotone solution. fit_pwl will
      determine whether to prefer a mono-up solution or a mono-down solution,
      unless min_slope or max_slope force a direction.
    min_slope: (None or float) Minimum slope between each adjacent pair of
      knots. Set to 0 for a monotone increasing solution.
    max_slope: (None or float) Maximum slope between each adjacent pair of
      knots. Set to 0 for a monotone decreasing solution.
    x_transform: (None or a strictly increasing 1D function): User-specified
      transform on x, to apply before piecewise-linear curve fitting. If None,
      fit_pwl chooses a transform using a heuristic. To specify fitting with no
      transform, pass in transform.identity.

  Returns:
    Tuple consisting of ([(x,y) knots], transform_fn). The curve performs linear
    interpolation in the transform_fn(x) space.
  """
  assert num_segments > 0, 'Cannot fit %d segment PWL' % num_segments
  assert num_samples > num_segments, (
      'num_samples must be at least num_segments + 1')

  x, y, w = sort_and_sample(x, y, w)
  if x_transform is None:
    x_transform = transform.find_best_transform(x, y, w)

  original_x = x
  trans_x = x_transform(x)
  assert np.isfinite(trans_x[[0, -1]]).all(), 'Transform must be defined on x.'

  # Pick a subset of x to use as candidate knots, and compress x, y, w around
  # those candidate knots.
  x_knots, x, y, w = (
      linear_condense.sample_condense_points(trans_x, y, w, num_samples))

  if mono:
    min_slope, max_slope = _get_mono_slope_bounds(y, w, min_slope, max_slope)

  # Fit a piecewise-linear curve in the transformed space.
  x_pnts, y_pnts = fit_pwl_points(x_knots, x, y, w, num_segments, min_slope,
                                  max_slope)

  # Recover the control point xs in the pre-transform space.
  x_pnts = original_x[trans_x.searchsorted(x_pnts)]
  if np.all(y_pnts == y_pnts[0]):  # The curve is constant.
    curve_points = [(x_pnts[0] - 1, y_pnts[0]), (x_pnts[0], y_pnts[0])]
  else:
    curve_points = list(zip(x_pnts, y_pnts))
  return curve_points, x_transform


def sort_and_sample(x, y, w, downsample_to=1e6):
  """Samples and sorts the data to fit a PWLCurve on.

  Samples each point with equal likelihood, once or zero times per point. Weight
  is not considered when sampling. For performance reasons, the precise number
  of final points is not guaranteed.

  Args:
    x: (numerical numpy array) The independent variable.
    y: (numerical numpy array) The dependent variable.
    w: (None or numerical numpy array) The weights of data points. Weights are
      NOT used in downsampling.
    downsample_to: (int or float) The approximate number of samples to take.

  Raises:
    ValueError: invalid input.

  Returns:
    A triple (sorted_x, y, w) of numpy arrays representing the dependent
    variable in sorted order, the independent variable, and the weights
    respectively.
  """
  x = np.array(x, copy=False)
  y = np.array(y, copy=False)
  if w is None:
    w = np.ones_like(x)
  else:
    w = np.array(w, copy=False)
    assert (w > 0).all(), 'Weights must be positive.'

  assert len(x) == len(y) == len(w) >= 1
  assert np.isfinite(x).all(), 'x-values must all be finite.'
  assert np.isfinite(y).all(), 'y-values must all be finite.'
  assert np.isfinite(w).all(), 'w-values must all be finite.'

  # Downsample to a manageable number of points to limit runtime.
  if len(x) > downsample_to * 1.01:
    np.random.seed(125)
    # Select each xyw with probability (downsample_to / len(x)) to yield
    # approximately downsample_to selections.
    fraction_kept = float(downsample_to) / len(x)
    mask = np.random.sample(size=len(x)) < fraction_kept
    x, y, w = x[mask], y[mask], w[mask]

  # Sort the points by x if any are out of order.
  if (x[1:] < x[:-1]).any():
    point_order = np.argsort(x)
    x, y, w = x[point_order], y[point_order], w[point_order]

  # Use float64 for precision.
  x = x.astype(float, copy=False)
  y = y.astype(float, copy=False)
  w = w.astype(float, copy=False)

  # Fuse points with the same x, so that all xs become unique.
  x, y, w = utils.fuse_sorted_points(x, y, w)
  return x, y, w


def _get_mono_slope_bounds(y, w, min_slope, max_slope):
  """Adjusts the slope bounds to impose monotonicity in the ideal direction.

  Args:
    y: (numpy array of floats) dependent variable.
    w: (numpy array of floats) the weights on data points.
    min_slope: (None or float) Minimum slope between adjacent pairs of knots.
    max_slope: (None or float) Maximum slope between adjacent pairs of knots.

  Returns:
    Pair (min_slope, max_slope) such that monotonicity is imposed. Either
    min_slope >= 0, or max_slope <= 0.
  """
  if ((min_slope is not None and min_slope >= 0) or
      (max_slope is not None and max_slope <= 0)):
    # The slope restrictions already impose monotonicity.
    return min_slope, max_slope

  # Calculate the preferable direction using isotonic regression.
  is_increasing = _is_increasing(y, w)

  # Update the slope restrictions.
  if is_increasing:
    min_slope = 0 if min_slope is None else max(0, min_slope)
  else:
    max_slope = 0 if max_slope is None else min(0, max_slope)

  return min_slope, max_slope


def _is_increasing(y, w):
  """Returns whether a mono-up or a mono-down sequence better approximates y.

  Tries to monotonize values in both the increasing and decreasing directions
  with respect to the given norm. Then returns whether the increasing or the
  decreasing sequence gives the better fit, as measured by mean squared error.

  Args:
    y: (numpy array of floats) sequence to monotonize.
    w: (numpy array of floats) weights.

  Returns:
    True if a monotonically increasing sequence is an equal or better fit to the
    data than a monotonically decreasing sequence. False otherwise.
  """
  increasing = isotonic.isotonic_regression(y, w, increasing=True)
  decreasing = isotonic.isotonic_regression(y, w, increasing=False)

  increasing_norm = np.average((increasing - y)**2, weights=w)
  decreasing_norm = np.average((decreasing - y)**2, weights=w)
  return increasing_norm <= decreasing_norm


def fit_pwl_points(x_knots,
                   x,
                   y,
                   w,
                   num_segments,
                   min_slope=None,
                   max_slope=None):
  """Fits a num_segments segment PWL to the sample points.

  Args:
    x_knots: (numpy array of floats) X-axis knot candidates. Must be unique, and
      sorted in ascending order.
    x: (numpy array of floats) X-axis values for minimizing mean squared error.
      Must be unique, and sorted in ascending order.
    y: (numpy array of floats) Y-axis values corresponding to x, for minimizing
      mean squared error.
    w: (numpy array of floats) Weights for each (x, y) pair.
    num_segments: (int) Maximum number of segments to fit.
    min_slope: (float) Minimum slope between each adjacent pair of knots. Set to
      0 for a monotone increasing solution, or None for no restriction.
    max_slope: (float) Maximum slope between each adjacent pair of knots. Set to
      0 for a monotone decreasing solution, or None for no restriction.

  Returns:
    Returns two tuple (x_points, y_points) where x_points (y_points) is the list
    of x-axis (y-axis) knot points of the fit PWL curve.
  """
  assert len(x) == len(y) == len(w) >= 1
  assert num_segments >= 1
  assert min_slope is None or max_slope is None or min_slope <= max_slope

  if len(x_knots) == 1 or np.all(y == y[0]):  # Constant function.
    y_mean = np.average(y, weights=w)
    return [x[0] - 1, x[0]], [y_mean, y_mean]

  solver = _WeightedLeastSquaresPWLSolver(x, y, w, min_slope, max_slope)
  return _fit_pwl_approx(x_knots, solver.solve, num_segments)


def _fit_pwl_approx(x_knots, solve_fn, num_segments):
  """Heuristic search for the best combination of knot xs and their y-values.

  Args:
    x_knots (list of floats): Available values to choose knot xs from.
    solve_fn (function): Function that takes a list of chosen knots and returns
      the tuple (best y-values for those knots, error with those values).
      _fit_pwl_approx searches for the combination of knot xs that minimizes
      solve_fn's error.
    num_segments (int): Number of segments to fit.

  Returns:
    Return a tuple (best_found_knot_xs, best_found_knot_ys), where the number of
    knots is one more than num_segments.
  """
  if len(x_knots) <= num_segments + 1:
    # There's no need to select knot-xs: just use all of them.
    knot_values, _ = solve_fn(x_knots)
    return x_knots, knot_values

  solve_cache = {}

  def add_one_point(cur_knots):
    """Exhaustive search for the best new knot point."""
    best_value, best_pnt, best_knot_values = np.inf, -1, None
    for pnt in x_knots:
      if pnt not in cur_knots:
        sorted_pnts = tuple(sorted(cur_knots + [pnt]))
        if sorted_pnts not in solve_cache:
          solve_cache[sorted_pnts] = solve_fn(sorted_pnts)
        knot_values, value = solve_cache[sorted_pnts]
        if value <= best_value:
          best_value, best_pnt, best_knot_values = value, pnt, knot_values

    assert best_value != np.inf
    cur_knots = sorted(cur_knots + [best_pnt])
    return cur_knots, list(best_knot_values), best_value

  best_value, best_knots, best_knot_values = (np.inf, [x_knots[0]], None)
  for _ in range(num_segments):
    best_knots, best_knot_values, best_value = add_one_point(best_knots)

  # Update each interior knot in the context of the other selected knots.
  for _ in range(10):
    knots = list(best_knots)
    for knot in knots:
      best_knots.remove(knot)
      best_knots, best_knot_values, value = add_one_point(best_knots)

    # As long as there's an improvement in fit, keep iterating.
    if value >= best_value:
      break
    best_value = value

  return best_knots, best_knot_values


class _WeightedLeastSquaresPWLSolver(object):
  """Computes the weighted least squares PWL knot ys for a fixed set of knot xs.

  Ex:
    solver = _WeightedLeastSquaresPWLSolver(x_pnts, y_pnts, weights)
    values_at_knots, sum_of_residuals = solver.solve(knot_xs)

    monoup_solver = _WeightedLeastSquaresPWLSolver(
        x_pnts, y_pnts, weights, min_slope=0)
    values_at_knots, sum_of_residuals = monoup_solver.solve(knot_xs)

  This solver uses weighted linear least squares to derive the optimal knot ys
  for a given set of knot xs, in order to minimize squared error over a provided
  set of (x,y,weight) points. Optionally, it can also bound the slope from each
  knot_y to the next. Slope bounds are most commonly used to impose
  monotonicity, by specifying min_slope = 0 (mono increasing) or max_slope = 0
  (mono decreasing).
    In general, linear_least_squares(A, b) finds v to minimize ||b - Av||^2,
  which is the mean squared error of using v to predict b from A.
  For our purposes:
    b is the vector of ground truth y_values, which we're given.
    v is the vector (y0, y1 - y0, y2 - y1, ...) of delta_knot_ys, which we find
    using the solver. Then knot_ys = np.cumsum(v).
    We solve for delta_knot_ys instead of knot_ys because it's simpler to bound
  delta_knot_ys than to directly impose slope restrictions on knot_ys.
  (Note that y0 is not actually a delta, and is deliberately not constrained.)
    A is a two-dimensional matrix such that A_ij represents the weight that the
  ith point in the data puts on the jth delta_knot_y. Consequently, Av simulates
  utils.eval_pwl_curve(ground_truth_x, zip(knot_xs, knot_ys)). For a PWLCurve,
  the value at any given x is the y-value of the previous knot plus a fraction
  of the delta_y to the next knot. (That fraction is
  (x - prev_knot_x) / (next_knot_x - prev_knot_x)).
    Suppose knot_k-1 <= x_i <= knot_k. Then the PWLCurve at x_i is
  knot_y_k-1 + frac * delta_knot_y_k = sum(v[:k]) + frac * v[k]. So A_ik = frac,
  A_ij = 1.0 for all j < k, and A_ij = 0.0 for all j > k.
    Because PWLCurves clamp their extremes, any x_i smaller than the least knot
  will have A_i0 = 1.0 and all other A_ij = 0.0. Likewise, any x_i larger than
  the greatest knot will have all A_ij = 1.0.
    A is uniquely determined by the x_values and the knot_xs. For example, if
  x_values are [0, 1, 2, 3, 4, 5, 6] and knot_xs are [1, 3, 6], then A will be
  the 7x3 matrix:

  (1    0    0)
  (1    0    0)
  (1   .5    0)
  (1    1    0)
  (1    1  .33)
  (1    1  .67)
  (1    1    1)

  To account for weights, we scale each entry in y_values and each row in A by
  the square root of the weight for the corresponding point. This scales each
  error term in (y_values - A * knot_delta_ys) by sqrt(weight), so that the mean
  squared error ||y_values - A * knot_delta_ys||^2 scales the squared error at
  each point by sqrt(weight)^2, which is just the weight at that point.

  _WeightedLeastSquaresPWLSolver is implemented as a class rather than a
  function so that we can precompute some values that are constant across calls
  to speed up each call. We usually call solve() on many different combinations
  of knot_xs for the same x_values, y_values, weights, and slope restrictions.
  """

  def __init__(self, x, y, w, min_slope=None, max_slope=None):
    """Constructor.

    Args:
      x: (numpy array of floats) X-axis values for minimizing mean squared
        error. Must be unique, and sorted in ascending order.
      y: (numpy array of floats) Y-axis values corresponding to x, for
        minimizing mean squared error.
      w: (numpy array of floats) Weights for each (x, y) pair.
      min_slope: float indicating the minimum slope between each adjacent pair
        of knots. Set to 0 to impose a monotone increasing solution.
      max_slope: float indicating the maximum slope between each adjacent pair
        of knots. Set to 0 to impose a monotone decreasing solution.
    """
    assert len(x) == len(y) == len(w) >= 1
    assert (w >= 0).all(), 'weights cannot be negative.'
    assert min_slope is None or max_slope is None or min_slope <= max_slope

    sqrt_w = np.sqrt(w, dtype=float)
    self._sqrt_w = sqrt_w.reshape(len(w), 1)
    self._weighted_y = np.array(y, dtype=float, copy=False) * sqrt_w
    self._x = np.array(x, dtype=float, copy=False)
    self._min_slope = min_slope
    self._max_slope = max_slope

  def _get_weighted_matrix(self, knot_xs):
    """Computes the matrix 'A' in ||b - Av||^2, weighted by _weight_matrix.

    Args:
      knot_xs: (float list or numpy array) X coordinates of current knot points.
        Must be unique and in ascending order.

    Returns:
      Two-dimensional matrix A such that A_ij represents the weight on the ith
      point in the data, multiplied by the weight that the ith point puts on the
      jth knot's delta y value.
    """
    knot_xs = list(map(float, knot_xs))  # float is faster than np.float64.
    # First we build the matrix for 'A' in A * knot_delta_ys ~= y_values.
    # For numpy vectorization speed, we use one row per knot and one column per
    # x-value during setup, and then take the transpose to get one row per x
    # value and one column per knot. This is faster because it allows us to fill
    # continuous blocks in the matrix's memory.
    width = len(self._x)
    height = len(knot_xs)
    assert height <= width, (
        'Solve() is underdetermined with more knots than points.')
    matrix = np.zeros(width * height, dtype=float)
    knot_indices_in_xs = np.searchsorted(self._x, knot_xs)

    # For x_i < knot_xs[0], A_i0 = 1.0 and all other A_ij = 0.0.
    matrix[0:knot_indices_in_xs[0]] = 1.0

    # For x_i >= knot_xs[j], A_ij = 1.0.
    for row, index in enumerate(knot_indices_in_xs):
      matrix[row * width + index:(row + 1) * width] = 1.0

    # For knot_xs[j-1] <= x_i < knot_xs[j],
    # A_ij = (x_i - knot_xs[j-1]) / (knot_xs[j] - knot_xs[j-1]).
    for index in range(1, height):
      lower, upper = knot_indices_in_xs[index - 1], knot_indices_in_xs[index]
      between_xs = self._x[lower:upper]

      lower_knot = knot_xs[index - 1]
      upper_knot = knot_xs[index]

      upper_knot_weight = (between_xs - lower_knot) / (upper_knot - lower_knot)
      upper_row = width * index
      matrix[upper_row + lower:upper_row + upper] = upper_knot_weight

    # Apply the weights once the matrix is full.
    matrix = matrix.reshape((height, width)).T
    weighted_matrix = matrix * self._sqrt_w
    return weighted_matrix

  def _get_bounds_on_y(self, knot_xs):
    """Computes the upper and lower bounds on the delta-ys between knots."""
    knot_xs = np.array(knot_xs, copy=False)
    delta_knot_xs = knot_xs[1:] - knot_xs[:-1]

    # The first y is a bias, not a delta, so it's never constrained.
    # Each delta_y satisfies min_slope <= delta_y / delta_x <= max_slope.
    if self._min_slope is not None:
      lower_bounds = [-np.inf] + list(self._min_slope * delta_knot_xs)
    else:
      lower_bounds = [-np.inf] * len(knot_xs)  # No lower bound.
    if self._max_slope is not None:
      upper_bounds = [np.inf] + list(self._max_slope * delta_knot_xs)
    else:
      upper_bounds = [np.inf] * len(knot_xs)  # No upper bound.

    return lower_bounds, upper_bounds

  def solve(self, knot_xs):
    """Computes the weighted least squares PWL knot ys for the given knot xs.

    Args:
      knot_xs: (float list or numpy array) X coordinates of current knot points.
        Must be unique and in ascending order. To avoid an underdetermined
        system, there cannot be more knot_xs than x. Also, there can be at most
        one knot that's >= all x, at most one knot that's <= all x, and at most
        two knots between two adjacent x. (Otherwise, the knots are
        underdetermined.)

    Returns:
      PWL function values at the knot points and sum of residuals.
    """
    weighted_matrix = self._get_weighted_matrix(knot_xs)

    # When unconstrained, use unconstrained linear least squares via numpy.
    if self._min_slope is None and self._max_slope is None:
      solution = np.linalg.lstsq(weighted_matrix, self._weighted_y, rcond=None)
      knot_ys = np.cumsum(solution[0])
      squared_error = solution[1][0] if solution[1].size != 0 else 0.0
      return knot_ys, squared_error

    # Bounded linear least squares via scipy.
    solution = scipy.optimize.lsq_linear(
        weighted_matrix,
        self._weighted_y,
        bounds=self._get_bounds_on_y(knot_xs),
        lsq_solver='exact',
        method='bvls',
    )
    knot_ys = np.cumsum(solution.x)  # Values are deltas --> cumsum.
    squared_error = 2 * solution.cost  # scipy uses half MSE for cost.
    return knot_ys, squared_error
