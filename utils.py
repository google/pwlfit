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


"""Utilities for PWLFit."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def fuse_sorted_points(sorted_xs, ys, ws):
  """For each unique x in the array sorted_xs, fuse all points with that x.

  This fusion changes the mean squared error, but preserves the delta MSE
  between any two functions of x.

  Args:
    sorted_xs: (numpy array) The dependent variable in sorted order. We fuse all
        points with the same value in sorted_xs.
    ys: (numpy array) The independent variable.
    ws: (numpy array) the weights on data points.

  Returns:
    A tuple (X, Y, W) of numpy arrays, where X is the unique values of sorted_xs
    in sorted order, Y is the weighted average of the ys corresponding to each x
    in X, and W is the sum of the ws corresponding to each x in X.
  """
  assert len(sorted_xs) == len(ys) == len(ws)
  is_unique = np.ones(len(sorted_xs), dtype=bool)
  np.not_equal(sorted_xs[1:], sorted_xs[:-1], out=is_unique[1:])
  if is_unique.all():  # All unique --> nothing to fuse.
    return sorted_xs, ys, ws

  unique_indices, = is_unique.nonzero()
  lower_bounds = unique_indices
  x_fuses = sorted_xs[lower_bounds]
  w_sums = np.add.reduceat(ws, lower_bounds)
  y_averages = np.add.reduceat(ys * ws, lower_bounds) / w_sums

  return x_fuses, y_averages, w_sums


def unique_on_sorted(sorted_a):
  """Computes unique elements from the given sorted array.

  O(n) version of np.unique(sorted_a) that only works on pre-sorted arrays.

  Args:
    sorted_a: (np.ndarray) a sorted 1d-array of points.

  Returns:
    An array consisting of the unique elements of sorted_a in sorted order.
  """
  is_unique = np.ones(len(sorted_a), dtype=bool)
  np.not_equal(sorted_a[1:], sorted_a[:-1], out=is_unique[1:])
  return sorted_a[is_unique]


def count_uniques_on_sorted(sorted_a):
  """Computes the number of unique elements in a sorted array.

  Args:
    sorted_a: (np.ndarray) a sorted 1d-array of points.

  Returns:
    The number of unique elements in sorted_a.
  """
  if len(sorted_a) < 1:
    return 0
  return 1 + np.count_nonzero(sorted_a[1:] != sorted_a[:-1])


def eval_pwl_curve(xs, curve_points, transform_fn=None):
  """Evaluate a given PWLCurve on data.

  Args:
    xs: (numpy array) Data to evaluate the curve on.
    curve_points: (list of pairs) The (x,y) control points of the PWLCurve. The
        xs must be unique and in ascending order.
    transform_fn: (function on numpy array) Function to map the space of xs
        before interpolating. Applied to the xs of the input and to the xs of
        the control points. Must be defined on the xs of the control points.

  Returns:
    A numpy array of the PWLCurve's value at each x in xs.
  """
  assert len(curve_points) >= 2, 'A PWLCurve must have at least two knots.'
  curve_xs, curve_ys = zip(*curve_points)
  curve_xs, curve_ys = np.asarray(curve_xs), np.asarray(curve_ys)
  assert len(set(curve_xs)) == len(curve_xs), 'Curve knot xs must be unique.'
  assert (np.sort(curve_xs) == curve_xs).all(), 'Curve knot xs must be ordered.'

  # Clamp the inputs to the range of the control points.
  xs = np.clip(xs, curve_xs[0], curve_xs[-1])
  if transform_fn is not None:
    xs = transform_fn(xs)
    curve_xs = transform_fn(curve_xs)

  indices = curve_xs.searchsorted(xs)

  # Extend curve for convenient boundary handling.
  curve_xs = np.concatenate([[curve_xs[0] - 1], curve_xs])
  curve_ys = np.concatenate([[curve_ys[0]], curve_ys])

  prev_x, prev_y = curve_xs[indices], curve_ys[indices]
  next_x, next_y = curve_xs[indices + 1], curve_ys[indices + 1]
  gap = next_x - prev_x

  return next_y * ((xs - prev_x) / gap) + prev_y * ((next_x - xs) / gap)
