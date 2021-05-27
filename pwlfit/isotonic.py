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

from typing import Optional, Sequence, Tuple
import numpy as np
from pwlfit import utils


def isotonic_regression(sequence: Sequence[float],
                        weights: Optional[Sequence[float]] = None,
                        increasing: bool = True) -> np.ndarray:
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
  if len(sequence) <= 1:
    return sequence

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

  def __init__(self, array: Sequence[float], weights: Sequence[float],
               index: int):
    """Class constructor."""
    self.start = index
    self.end = index + 1
    self.sum = array[index] * weights[index]
    self.weight_sum = weights[index]

  def merge_with_next_block(self, right: '_Block') -> None:
    """Merges this block with the block immediately to the right of it. O(1).

    Args:
      right: Block to merge with. Must be immediately to the right of this block
        (i.e., self._end == right._start).
    """
    # Blocks must be adjacent.
    assert self.end == right.start

    self.sum += right.sum
    self.weight_sum += right.weight_sum
    self.end = right.end

  def merge_and_return_error(self, right: '_Block') -> float:
    """Merge with the block immediately to the right and return the error. O(1).

    Merging two blocks produces a single block whose value is the weighted
    average of the original values and whose total weight is the sum of their
    total weights. However, if the values of the original blocks differ, the
    merge is lossy and produces a squared error.

    Args:
      right: Block to merge with. Must be immediately to the right of this block
        (i.e., self._end == right._start).

    Returns:
      Squared error of using the merged block's value to predict the values of
      the original two blocks.
    """
    # Blocks must be adjacent.
    assert self.end == right.start

    # Squared error from merging two blocks. Zero if their values are the same.
    error = ((self.value() - right.value())**2 *
             self.weight_sum * right.weight_sum / (
                 self.weight_sum + right.weight_sum))

    self.sum += right.sum
    self.weight_sum += right.weight_sum
    self.end = right.end
    return error

  def length(self) -> int:
    return self.end - self.start

  def value(self) -> float:
    return self.sum / self.weight_sum


def _pool_adjacent_violators(sequence: Sequence[float],
                             weights: Sequence[float]) -> np.ndarray:
  """Implements the Pool Adjacent Violators algorithm for isotonic regression.

  Args:
    sequence: (sequence of floats) sequence to monotonize.
    weights: (sequence of floats) weights to use for the elements in sequence.

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


def _isotonic_partial_errors(sequence: Sequence[float],
                             weights: Sequence[float]) -> np.ndarray:
  """Finds the partial errors of isotonic regression over subsequences. O(n).

  Pool adjacent violators, but we track the squared error at each step. This is
  equivalent to performing isotonic regression separately on each sequence[:i],
  except this approach is O(n) instead of O(n^2).

  Args:
    sequence: (sequence of floats) sequence to monotonize.
    weights: (sequence of floats) weights to use for the elements in sequence.

  Returns:
    Numpy array 'errors' of len(sequence) + 1 floats, where errors[i] is the
    weighted squared error of isotonic regression over sequence[:i].
  """
  error = 0
  errors = np.zeros(len(sequence) + 1)
  blocks = [_Block(sequence, weights, 0)]
  for index in range(1, len(sequence)):
    prev_block = blocks[-1]
    cur_block = _Block(sequence, weights, index)

    # Iteratively merge out of order blocks.
    while prev_block and prev_block.value() > cur_block.value():
      error += prev_block.merge_and_return_error(cur_block)
      cur_block = prev_block
      blocks.pop()
      prev_block = None if not blocks else blocks[-1]

    blocks.append(cur_block)
    errors[index + 1] = error  # errors[i] is error of merging first i blocks.

  return errors


def bitonic_peak_and_error(sequence: Sequence[float],
                           weights: Sequence[float]) -> Tuple[int, float]:
  """Finds the index of the ideal bitonic peak to minimize squared error. O(n).

  Uses prefix isotonic regression as described in
  https://web.eecs.umich.edu/~qstout/pap/UniRegRevised.pdf to find the index of
  the peak that minimizes the error of a concave-down bitonic regression.

  Args:
    sequence: (sequence of floats) sequence to monotonize.
    weights: (sequence of floats) weights to use for the elements in sequence.

  Returns:
    Tuple: (index of the bitonic peak, mse of bitonic regression using that
    peak.)
  """
  # We determine the error of forcing sequence[:i] to be increasing and
  # sequence[i:] to be decreasing for each i in [0...n]. Then we pick the i with
  # the lowest combined error.
  prefix_errors = _isotonic_partial_errors(sequence, weights)
  suffix_errors = _isotonic_partial_errors(sequence[::-1], weights[::-1])
  combined_errors = prefix_errors + suffix_errors[::-1]
  peak_index = np.argmin(combined_errors)
  return peak_index, combined_errors[peak_index]


def bitonic_regression(sequence, weights=None, convex=True):
  """Returns a bitonic sequence which most closely approximates sequence.

  Args:
    sequence: List of numbers to approximate.
    weights: List of positive weights to use.
    convex: If True, the sequence should be convex, otherwise concave. Defaults
            to True.
  Returns:
    A bitonic sequence approximating the given sequence. Note that the output
    of this function is always a sequence of floats.
  """
  sequence = np.array(sequence, copy=False, dtype=float)
  if len(sequence) <= 1:
    return sequence

  if convex:
    return -bitonic_regression(-sequence, weights, convex=False)

  if weights is None:
    weights = np.ones_like(sequence)
  else:
    weights = np.array(weights, copy=False, dtype=float)
    utils.expect(len(weights) == len(sequence),
                 'Weights must be same size as sequence.')
    utils.expect((weights > 0).all(), 'Weights must be positive.')

  index, _ = bitonic_peak_and_error(sequence, weights)
  best_prefix = isotonic_regression(
      sequence[:index], weights[:index], increasing=True)
  best_suffix = isotonic_regression(
      sequence[index:], weights[index:], increasing=False)

  return np.concatenate([best_prefix, best_suffix])
