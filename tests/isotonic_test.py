"""Tests for google3.quality.ranklab.main.optimization.isotonic_regression."""

import unittest
from absl.testing import parameterized

import numpy as np
from pwlfit import isotonic
from pwlfit import test_util


class IsotonicRegressionTest(test_util.PWLFitTest, parameterized.TestCase):

  @parameterized.named_parameters(
      # Preserves monotonic sequences when the direction matches.
      ('increasing', [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], True),
      ('decreasing', [4, 3, 2, 1, 0], [4, 3, 2, 1, 0], False),
      # Forces a monotone solution when the input isn't monotone.
      ('increasing on non-mono', [0, 1, 2, 100, 4], [0, 1, 2, 52, 52], True),
      ('increasing on non-mono weighted #1', [0, 1, 2, 100, 4],
       [0, 1, 2, 68, 68], True, [1, 1, 1, 2, 1]),
      ('decreasing on non-mono', [4, 100, 2, 1, 0], [52, 52, 2, 1, 0], False),
      ('decreasing on non-mono weighted', [4, 100, 2, 1, 0], [68, 68, 2, 1, 0],
       False, [1, 2, 1, 1, 1]),
      ('increasing on non-mono weighted #2', [0, 1, 2, 100, 4, 4],
       [0, 1, 2, 61.6, 61.6, 61.6], True, [1, 1, 1, 3, 1, 1]),
      ('decreasing on non-mono weighted #2', [4, 4, 100, 2, 1, 0],
       [61.6, 61.6, 61.6, 2, 1, 0], False, [1, 1, 3, 1, 1, 1]),
      ('increasing range', list(range(100)), list(range(100)), True),
      ('decreasing range', list(range(100)), [49.5] * 100, False),
      # Single-point input shouldn't be affected.
      ('one-point increasing', [10], [10], True),
      ('one-point decreasing', [10], [10], False),
      # Verify that we can pass the data as numpy arrays.
      ('numpy array input', np.arange(100), np.arange(100), True, np.ones(100)),
      )
  def test_isotonic_regression(self,
                               input_seq,
                               expected_seq,
                               increasing,
                               weights=None):
    mono_seq = isotonic.isotonic_regression(input_seq, weights, increasing)
    np.testing.assert_array_equal(mono_seq, expected_seq)

    if increasing:
      self.assert_increasing(mono_seq)
    else:
      self.assert_decreasing(mono_seq)


class BitonicRegressionTest(test_util.PWLFitTest, parameterized.TestCase):

  @parameterized.named_parameters(
      # Preserves monotonic sequences.
      ('mono up convex', [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], True),
      ('mono up concave', [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], False),
      ('mono down convex', [4, 3, 2, 1, 0], [4, 3, 2, 1, 0], True),
      ('mono down concave', [4, 3, 2, 1, 0], [4, 3, 2, 1, 0], False),
      # Preserves bitonic sequences with the correct concavity.
      ('convex', [3, 1, 2, 3, 4], [3, 1, 2, 3, 4], True),
      ('concave', [1, 2, 3, 2, 1], [1, 2, 3, 2, 1], False),
      # Bitonic sequences with the wrong concavity get monotone approximations.
      ('concave regression on convex', [3, 1, 2, 3, 4], [2, 2, 2, 3, 4], False),
      ('convex regression on concave', [1, 2, 3, 2, 1], [1, 2, 2, 2, 2], True),
      # For non-bitonic sequences, bitonic regression introduces some error.
      ('non-bitonic #1', [3, 1, 2, 3, 2], [3, 1, 2, 2.5, 2.5], True),
      ('non-bitonic #2', [1, 2, 5, 2, 3], [1, 2, 5, 2.5, 2.5], False),
      # The regression prioritizes getting heavily weighted points right.
      ('weighted #1', [3, 1, 2, 3, 2], [3, 1, 2, 2, 2], True, [1, 1, 1, 1, 1e5],
       4),
      ('weighted #2', [1, 2, 5, 2, 3], [1, 2, 5, 3, 3], False,
       [1, 1, 1, 1, 1e5], 4),
      ('weighted #3', [2, 1, 2, 1, 2], [2, 1, 2, 2, 2], True,
       [1, 1e5, 1e5, 1, 1], 4),
      # Single-point input shouldn't be affected.
      ('one-point convex', [10], [10], True),
      ('one-point concave', [10], [10], False),
      # Verify that we can pass the data as numpy arrays.
      ('numpy array input', np.array([2, 1, 2, 1, 2]),
       np.array([2, 1, 2, 2, 2]), True, np.array([1, 1e5, 1e5, 1, 1]), 4),
      )
  def test_bitonic_regression(self,
                              input_seq,
                              expected_seq,
                              convex,
                              weights=None,
                              places=None):
    bitonic_seq = isotonic.bitonic_regression(
        input_seq, weights=weights, convex=convex)
    if not places:
      np.testing.assert_array_equal(bitonic_seq, expected_seq)
    else:
      for a, b in zip(bitonic_seq, expected_seq):
        self.assertAlmostEqual(a, b, places=places)

    # Count the number of times the curve changes direction. This should be at
    # most one.
    num_inversions = 0
    for a, b, c in zip(bitonic_seq[:-2], bitonic_seq[1:-1], bitonic_seq[2:]):
      if (b - a) * (c - b) < 0:
        num_inversions += 1

    self.assertLessEqual(num_inversions, 1)

  @parameterized.named_parameters(
      # Preserves monotonic sequences.
      ('mono up', [0, 1, 2, 3, 4], 4, 0),
      ('mono down', [4, 3, 2, 1, 0], 0, 0),
      # Concave-down sequences should never produce errors.
      ('concave down', [1, 2, 3, 2, 1], 2, 0),
      # When multiple peaks are equally good, we favor the first ideal peak.
      ('multiple ideal peaks', [1, 2, 3, 3, 2, 1], 2, 0),
      # 1-2 element sequences should never produce errors.
      ('one point', [1], 0, 0),
      ('two-point increasing', [1, 2], 1, 0),
      ('two-point decreasing', [2, 1], 0, 0),
      # When the data isn't concave, some error is unavoidable.
      # The closest fit is [2, 2, 2, 3, 4].
      ('convex', [3, 1, 2, 3, 4], 4, 2),
      # The closest fit is [1, 2, 3, 2.5, 2.5].
      ('non-bitonic #1', [1, 2, 3, 2, 3], 2, 0.5),
      # The closest fit is [1, 2, 5, 2.5, 2.5].
      ('non-bitonic # 2', [1, 2, 5, 2, 3], 2, .5),
      # The heavily weighted value needs to match almost perfectly, so the
      # closest fit is approximately [1, 2, 5, 3, 3].
      ('weighted', [1, 2, 5, 2, 3], 2, 1, [1, 1, 1, 1, 100000], 4),
      ('numpy array input', np.array([1, 2, 5, 2, 3]), 2, 1,
       np.array([1, 1, 1, 1, 100000]), 4),
      )
  def test_bitonic_peak_and_error(self,
                                  input_seq,
                                  expected_peak,
                                  expected_error,
                                  weights=None,
                                  places=10):
    if weights is None:
      weights = np.ones_like(input_seq)
    peak, error = isotonic.bitonic_peak_and_error(input_seq, weights)
    self.assertEqual(expected_peak, peak)
    self.assertAlmostEqual(expected_error, error, places=places)


if __name__ == '__main__':
  unittest.main()
