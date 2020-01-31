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


"""Isotonic regression via pool adjacent violators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pwlfit import utils
from six.moves import range


def isotonic_regression(sequence, weights=None, increasing=True):
  """Returns a monotonic sequence which most closely approximates sequence.

  Args:
    sequence: (numpy array of floats) sequence to monotonize.
    weights: (numpy array of floats) weights to use for the elements in
      sequence.
    increasing: (bool) True if the sequence should be monotonically increasing,
      otherwise decreasing. Defaults to True.

  Returns:
    A monotonic array of floats approximating the given sequence.
  """
  sequence = np.array(sequence, copy=False, dtype=float)

  if not increasing:
    return -isotonic_regression(-sequence, weights, True)

  if weights is None:
    weights = np.ones_like(sequence)
  else:
    weights = np.array(weights, copy=False, dtype=float)
    utils.expect(
        len(weights) == len(sequence), 'Weights must be same size as sequence.')
    utils.expect((weights > 0).all(), 'Weights must be positive.')

  return _pool_adjacent_violators(sequence, weights)


class _Block(object):
  """Represents a contiguous subarray of the provided array."""

  def __init__(self, array, weights, index):
    """Class constructor."""
    self.start = index
    self.end = index + 1
    self.sum = array[index] * weights[index]
    self.weight_sum = weights[index]

  def merge_with_next_block(self, right):
    """Merges this block with the block immediately to the right of it.

    O(1).

    Args:
      right: Block to merge with. Must be immediately to the right of this block
        (i.e., self._end == right._start).
    """
    # Blocks must be adjacent.
    assert self.end == right.start

    self.sum += right.sum
    self.weight_sum += right.weight_sum
    self.end = right.end

  def length(self):
    return self.end - self.start

  def value(self):
    return self.sum / self.weight_sum


def _pool_adjacent_violators(sequence, weights):
  """Implements the Pool Adjacent Violators algorithm for isotonic regression.

  Args:
    sequence: (numpy array of floats) sequence to monotonize.
    weights: (numpy array of floats) weights to use for the elements in
      sequence.

  Returns:
    Numpy array of floats which is the closest increasing monotonic match to the
    given sequence under the weighted L2 norm.
  """
  blocks = [_Block(sequence, weights, 0)]
  for index in range(1, len(sequence)):
    prev_block = blocks[-1]
    cur_block = _Block(sequence, weights, index)

    # Iteratively merge out of order blocks.
    while prev_block and prev_block.value() > cur_block.value():
      prev_block.merge_with_next_block(cur_block)
      cur_block = prev_block
      blocks.pop()
      prev_block = None if not blocks else blocks[-1]

    blocks.append(cur_block)

  values = [block.value() for block in blocks]
  lengths = [block.length() for block in blocks]
  return np.repeat(values, lengths)
