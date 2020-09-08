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

from typing import Tuple
import numpy as np


def fuse_sorted_points(
    sorted_xs: np.ndarray, ys: np.ndarray,
    ws: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
  expect(len(sorted_xs) == len(ys) == len(ws))
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


def unique_on_sorted(sorted_a: np.ndarray) -> np.ndarray:
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


def count_uniques_on_sorted(sorted_a: np.ndarray) -> int:
  """Computes the number of unique elements in a sorted array.

  Args:
    sorted_a: (np.ndarray) a sorted 1d-array of points.

  Returns:
    The number of unique elements in sorted_a.
  """
  if len(sorted_a) < 1:
    return 0
  return 1 + np.count_nonzero(sorted_a[1:] != sorted_a[:-1])


def expect(condition: bool, message: str = '') -> None:
  """Raises a ValueError if condition isn't truthy.

  Args:
    condition: (boolean) Whether or not to raise a ValueError.
    message: (optional str) The message raised with the error.

  Raises:
    ValueError: [message]
  """
  if not condition:
    raise ValueError(message)
