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

"""Tests for pwlfit/utils."""
import unittest

import numpy as np
from pwlfit import test_util
from pwlfit import utils


def _best_fit_line(x, y, w):
  sqrt_w = np.sqrt(w)
  matrix = np.vstack([x * sqrt_w, sqrt_w]).T
  slope, intercept = np.linalg.lstsq(matrix, y * sqrt_w, rcond=None)[0]
  return slope, intercept


class FuseSortedPointsTest(test_util.PWLFitTest):

  def test_fuse_sorted_points_on_example(self):
    x = np.array([0., 0., 1., 1., 1., 2.])
    y = np.array([1., 4., 2., 3., 4., 5.])
    w = np.array([2., 1., 1., 1., 1., 2.])
    fused_x, fused_y, fused_w = utils.fuse_sorted_points(x, y, w)
    self.assert_allclose(np.array([0., 1., 2.]), fused_x)
    self.assert_allclose(np.array([2., 3., 5.]), fused_y)
    self.assert_allclose(np.array([3., 3., 2.]), fused_w)

  def test_fuse_sorted_points_does_nothing_on_unique_xs(self):
    np.random.seed(954345)
    x = np.arange(100)
    y = np.random.normal(size=100)
    w = np.random.uniform(size=100)
    fused_x, fused_y, fused_w = utils.fuse_sorted_points(x, y, w)
    self.assert_allclose(x, fused_x)
    self.assert_allclose(y, fused_y)
    self.assert_allclose(w, fused_w)

  def test_fuse_sorted_points_maintains_totals(self):
    np.random.seed(954346)
    x = np.sort(np.random.randint(10, size=1000).astype(float))
    y = x + np.random.normal(size=1000)
    w = np.random.uniform(size=1000)
    fused_x, fused_y, fused_w = utils.fuse_sorted_points(x, y, w)
    np.testing.assert_equal(np.unique(x), fused_x)
    self.assertAlmostEqual(w.sum(), fused_w.sum())
    self.assertAlmostEqual((y * w).sum(), (fused_y * fused_w).sum())

  def test_fuse_sorted_points_maintains_best_line(self):
    np.random.seed(954346)
    x = np.sort(np.random.randint(10, size=1000).astype(float))
    y = x + np.random.normal(size=1000)
    w = np.random.uniform(size=1000)
    slope, intercept = _best_fit_line(x, y, w)

    fused_x, fused_y, fused_w = utils.fuse_sorted_points(x, y, w)
    fused_slope, fused_intercept = _best_fit_line(fused_x, fused_y, fused_w)
    self.assertAlmostEqual(slope, fused_slope)
    self.assertAlmostEqual(intercept, fused_intercept)


class UniqueOnSortedTest(test_util.PWLFitTest):

  def test_unique_on_sorted_with_all_values_unique(self):
    np.random.seed(4)
    x = np.sort(np.random.uniform(size=10**2))
    unique_x = np.unique(x)
    np.testing.assert_array_equal(unique_x, utils.unique_on_sorted(x))
    self.assertEqual(len(unique_x), utils.count_uniques_on_sorted(x))

  def test_unique_on_sorted_with_repeated_values(self):
    np.random.seed(5)
    x = np.sort(np.random.randint(10, size=10**2))
    unique_x = np.unique(x)
    np.testing.assert_array_equal(unique_x, utils.unique_on_sorted(x))
    self.assertEqual(len(unique_x), utils.count_uniques_on_sorted(x))

  def test_unique_on_sorted_on_empty(self):
    x = np.array([], dtype=float)
    unique_x = np.unique(x)
    np.testing.assert_array_equal(unique_x, utils.unique_on_sorted(x))
    self.assertEqual(len(unique_x), utils.count_uniques_on_sorted(x))


class ExpectTest(test_util.PWLFitTest):

  def test_expect_does_nothing_when_true(self):
    utils.expect(True)
    utils.expect(True, 'True should be True.')

  def test_expect_raises_when_false(self):
    with self.assertRaises(ValueError):
      utils.expect(False)

    with self.assertRaisesRegex(ValueError, 'Value is False'):
      utils.expect(False, 'Value is False.')


if __name__ == '__main__':
  unittest.main()
