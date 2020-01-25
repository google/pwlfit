# Lint as: python2, python3
"""Tests for google3.quality.ranklab.main.optimization.isotonic_regression."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import unittest

import numpy as np
import isotonic
import test_util
from six.moves import range


class IsotonicRegressionTest(test_util.PWLFitTest):

  def assert_monotized_correctly(self,
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

  def test_isotonic_regression(self):
    # We should always reproduce monotonic sequences
    self.assert_monotized_correctly([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], True)
    self.assert_monotized_correctly([4, 3, 2, 1, 0], [4, 3, 2, 1, 0], False)

    self.assert_monotized_correctly([0, 1, 2, 100, 4], [0, 1, 2, 52, 52], True)
    self.assert_monotized_correctly([0, 1, 2, 100, 4], [0, 1, 2, 68, 68],
                                    True,
                                    weights=[1, 1, 1, 2, 1])

    self.assert_monotized_correctly([4, 100, 2, 1, 0], [52, 52, 2, 1, 0], False)
    self.assert_monotized_correctly([4, 100, 2, 1, 0], [68, 68, 2, 1, 0],
                                    False,
                                    weights=[1, 2, 1, 1, 1])

    self.assert_monotized_correctly([0, 1, 2, 100, 4, 4],
                                    [0, 1, 2, 61.6, 61.6, 61.6],
                                    True,
                                    weights=[1, 1, 1, 3, 1, 1])

    self.assert_monotized_correctly([4, 4, 100, 2, 1, 0],
                                    [61.6, 61.6, 61.6, 2, 1, 0],
                                    False,
                                    weights=[1, 1, 3, 1, 1, 1])

    self.assert_monotized_correctly(list(range(100)), list(range(100)), True)
    self.assert_monotized_correctly(list(range(100)), [49.5] * 100, False)


if __name__ == '__main__':
  unittest.main()
