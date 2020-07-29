# Lint as: python3
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
import six


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


class EvalPWLCurveTest(test_util.PWLFitTest):

  def test_eval(self):
    curve = [(1., 5.), (5., 13.), (10., 15.)]
    x = [0, 1, 2, 5, 7.5, 10, 20]
    expected_y = [5, 5, 7, 13, 14, 15, 15]
    predicted_y = utils.eval_pwl_curve(x, curve)
    self.assert_allclose(expected_y, predicted_y)

  def test_eval_with_exp_transform(self):
    curve = [(1., 5.), (5., 13.), (10., 15.)]
    x = [0, 1, 2, 5, 7.5, 10, 20]

    # Shift x to exponential space.
    curve_x, curve_y = zip(*curve)
    curve_x = np.exp(curve_x)
    exp_curve = list(zip(curve_x, curve_y))
    exp_x = np.exp(x)

    # Perform interpolation in the log-x space, counteracting the shift in x.
    expected_y = [5, 5, 7, 13, 14, 15, 15]
    predicted_y = utils.eval_pwl_curve(exp_x, exp_curve, np.log)
    self.assert_allclose(expected_y, predicted_y)


class ExpectTest(test_util.PWLFitTest):

  def test_expect_does_nothing_when_true(self):
    utils.expect(True)
    utils.expect(True, 'True should be True.')

  def test_expect_raises_when_false(self):
    with self.assertRaises(ValueError):
      utils.expect(False)

    with six.assertRaisesRegex(self, ValueError, 'Value is False'):
      utils.expect(False, 'Value is False.')


class RoundToSigFigsTest(test_util.PWLFitTest):

  def test_round(self):
    curve = [(1.234, 5.4321), (5.6789, 14.321)]
    rounded_curve = utils.round_to_sig_figs(curve, 2)
    expected_curve = [(1.2, 5.4), (5.7, 14)]
    self.assertEqual(expected_curve, rounded_curve)

  def test_round_ydigits(self):
    curve = [(1.234, 5.4321), (5.6789, 14.321)]
    rounded_curve = utils.round_to_sig_figs(curve, 2, 4)
    expected_curve = [(1.2, 5.432), (5.7, 14.32)]
    self.assertEqual(expected_curve, rounded_curve)

  def test_round_large_values(self):
    curve = [(1234, 54321), (56789, 14321)]
    rounded_curve = utils.round_to_sig_figs(curve, 2)
    expected_curve = [(1200, 54000), (57000, 14000)]
    self.assertEqual(expected_curve, rounded_curve)

  def test_round_no_change(self):
    curve = [(1., 5.), (5., 13.), (10., 15.)]
    rounded_curve = utils.round_to_sig_figs(curve, 2)
    self.assertEqual(curve, rounded_curve)

  def test_round_increases_figs_for_close_xs(self):
    curve = [(1.23456, 5.4321), (1.23467, 6.5432), (5.6789, 14.321)]
    rounded_curve = utils.round_to_sig_figs(curve, 2)
    expected_curve = [(1.2346, 5.4), (1.2347, 6.5), (5.6789, 14)]
    self.assertEqual(expected_curve, rounded_curve)

  def test_round_raises_value_error_for_repeat_xs(self):
    bad_curve = [(1., 5.), (1., 13.), (10., 15.)]
    with self.assertRaises(ValueError):
      utils.round_to_sig_figs(bad_curve, 2)


if __name__ == '__main__':
  unittest.main()
