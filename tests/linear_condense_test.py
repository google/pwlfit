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


"""Tests for pwlfit/linear_condense."""
import unittest

import numpy as np
from pwlfit import linear_condense
from pwlfit import pwlcurve
from pwlfit import test_util
from pwlfit import utils


def _line_mse_on_data(slope, intercept, x, y, w):
  x = np.array(x, copy=False)
  y = np.array(y, copy=False)
  w = np.array(w, copy=False)
  return np.sum((slope * x + intercept - y)**2 * w)


def _best_fit_line(x, y, w):
  sqrt_w = np.sqrt(w)
  matrix = np.vstack([x * sqrt_w, sqrt_w]).T
  slope, intercept = np.linalg.lstsq(matrix, y * sqrt_w, rcond=None)[0]
  return slope, intercept


def _curve_error_on_points(x, y, w, curve):
  return np.average((y - curve.eval(x))**2, weights=w)


class LinearCondenseTest(test_util.PWLFitTest):

  def test_recenter_at_zero_does_nothing_when_centroid_is_already_zero(self):
    x = np.array([-2., 0., 1.])
    y = np.array([1., -1., 1.])
    w = np.array([1., 3., 2.])
    centered_x, centered_y, cx, cy = linear_condense._recenter_at_zero(x, y, w)
    self.assert_allclose(x, centered_x)
    self.assert_allclose(y, centered_y)
    self.assert_allclose((0., 0.), (cx, cy))

  def test_recenter_at_zero(self):
    np.random.seed(954347)
    # Generate data with a nonzero centroid.
    x = np.random.normal(loc=3.0, size=1000)
    y = np.random.normal(loc=-1.0, size=1000)
    w = np.random.uniform(size=1000)
    centered_x, centered_y, cx, cy = linear_condense._recenter_at_zero(x, y, w)
    self.assert_allclose(x, centered_x + cx)
    self.assert_allclose(y, centered_y + cy)
    self.assertAlmostEqual(cx, 3.0, delta=.1)
    self.assertAlmostEqual(cy, -1.0, delta=.1)
    # Centroid of centered points should be ~(0,0).
    self.assertAlmostEqual(0., centered_x.dot(w) / w.sum())
    self.assertAlmostEqual(0., centered_y.dot(w) / w.sum())

  def test_linear_condense_does_nothing_on_fewer_than_three_points(self):
    x, y, w = np.array([]), np.array([]), np.array([])
    self.assert_allclose([x, y, w], linear_condense.linear_condense(x, y, w))

    x, y, w = np.array([1.]), np.array([2.]), np.array([.5])
    self.assert_allclose([x, y, w], linear_condense.linear_condense(x, y, w))

    x, y, w = np.array([1., 2.]), np.array([2., 3.]), np.array([.5, 4.])
    self.assert_allclose([x, y, w], linear_condense.linear_condense(x, y, w))

  def test_linear_condense_returns_centroid_when_only_one_unique_x(self):
    x = np.array([1., 1., 1.])
    y = np.array([2., 3., 8.])
    w = np.array([1., 2., 1.])
    centroid = ([1.], [4.], [4.])
    self.assert_allclose(centroid, linear_condense.linear_condense(x, y, w))

  def test_linear_condense_preserves_centroid(self):
    np.random.seed(954348)
    x = np.random.normal(loc=3.0, scale=2.0, size=100)
    y = x + np.random.normal(loc=-1.0, scale=4.3, size=100)
    w = np.random.uniform(size=100)

    # Prod's centroid: (cx, cy, w_sum).
    w_sum = w.sum()
    cx = np.dot(x, w) / w_sum
    cy = np.dot(y, w) / w_sum

    # Condensed points centroid.
    condensed_x, condensed_y, condensed_w = linear_condense.linear_condense(
        x, y, w)
    condensed_w_sum = sum(condensed_w)
    condensed_cx = np.dot(condensed_x, condensed_w) / condensed_w_sum
    condensed_cy = np.dot(condensed_y, condensed_w) / condensed_w_sum

    self.assert_allclose(cx, condensed_cx)
    self.assert_allclose(cy, condensed_cy)
    self.assert_allclose(w_sum, condensed_w_sum)

  def test_linear_condense_preserves_translation(self):
    np.random.seed(954348)
    x = np.random.normal(loc=3.0, scale=2.0, size=100)
    y = x + np.random.normal(loc=-1.0, scale=4.3, size=100)
    w = np.random.uniform(size=100)

    # Translate(condense(points)) = condense(translate(points)).
    # Note: translations in x and y and preserved, but NOT translations in w.
    x_trans, y_trans = np.random.normal(size=2)
    condensed_x, condensed_y, condensed_w = linear_condense.linear_condense(
        x, y, w)
    trans_condensed_x, trans_condensed_y, trans_condensed_w = (
        linear_condense.linear_condense(x + x_trans, y + y_trans, w))

    self.assert_allclose(condensed_x + x_trans, trans_condensed_x)
    self.assert_allclose(condensed_y + y_trans, trans_condensed_y)
    self.assert_allclose(condensed_w, trans_condensed_w)

  def test_linear_condense_preserves_scaling(self):
    np.random.seed(954348)
    x = np.random.normal(loc=3.0, scale=2.0, size=100)
    y = x + np.random.normal(loc=-1.0, scale=4.3, size=100)
    w = np.random.uniform(size=100)

    # Scale(condense(points)) ~= condense(scale(points)).
    # Scaling works for all of x, y, and w, provided w stays positive.
    x_scale = np.random.normal()
    y_scale = np.random.normal()
    w_scale = abs(np.random.normal())
    condensed_x, condensed_y, condensed_w = linear_condense.linear_condense(
        x, y, w)
    scaled_condensed_x, scaled_condensed_y, scaled_condensed_w = (
        linear_condense.linear_condense(x * x_scale, y * y_scale, w * w_scale))

    self.assert_allclose(np.array(condensed_x) * x_scale, scaled_condensed_x)
    self.assert_allclose(np.array(condensed_y) * y_scale, scaled_condensed_y)
    self.assert_allclose(np.array(condensed_w) * w_scale, scaled_condensed_w)

  def test_linear_condense_invariant_to_fusing_points(self):
    np.random.seed(954348)
    x = np.sort(np.random.randint(10, size=1000).astype(float))
    y = x + np.random.normal(size=1000)
    w = np.random.uniform(size=1000)

    condensed_x, condensed_y, condensed_w = linear_condense.linear_condense(
        x, y, w)
    fused_x, fused_y, fused_w = utils.fuse_sorted_points(x, y, w)
    fused_condensed_x, fused_condensed_y, fused_condensed_w = (
        linear_condense.linear_condense(fused_x, fused_y, fused_w))

    self.assert_allclose(condensed_x, fused_condensed_x)
    self.assert_allclose(condensed_y, fused_condensed_y)
    self.assert_allclose(condensed_w, fused_condensed_w)

  def test_linear_condense_preserves_best_fit_line(self):
    np.random.seed(954358)
    x = np.random.normal(loc=3.0, scale=2.0, size=100)
    y = x + np.random.normal(loc=-1.0, scale=4.3, size=100)
    w = np.random.uniform(size=100)

    condensed_x, condensed_y, condensed_w = linear_condense.linear_condense(
        x, y, w)
    condensed_slope, condensed_intercept = _best_fit_line(
        condensed_x, condensed_y, condensed_w)
    slope, intercept = _best_fit_line(x, y, w)

    self.assertAlmostEqual(slope, condensed_slope)
    self.assertAlmostEqual(intercept, condensed_intercept)

  def test_linear_condense_preserves_mse_diff_between_any_two_lines(self):
    np.random.seed(954349)
    x = np.random.normal(loc=3.0, scale=2.0, size=100)
    y = x + np.random.normal(loc=-1.0, scale=4.3, size=100)
    w = np.random.uniform(size=100)
    condensed_x, condensed_y, condensed_w = linear_condense.linear_condense(
        x, y, w)

    # Generate two random lines. Measure their mse on data and condensed data.
    slope1, intercept1 = np.random.normal(size=2)
    slope2, intercept2 = np.random.normal(size=2)
    l1_mse = _line_mse_on_data(slope1, intercept1, x, y, w)
    l2_mse = _line_mse_on_data(slope2, intercept2, x, y, w)
    condensed_l1_mse = _line_mse_on_data(slope1, intercept1, condensed_x,
                                         condensed_y, condensed_w)
    condensed_l2_mse = _line_mse_on_data(slope2, intercept2, condensed_x,
                                         condensed_y, condensed_w)

    # Condensed points squash noise, so l1_mse != condensed_l1_mse.
    # However, the difference true_mse(line) - condensed_mse(line) should be
    # approximately equal for every any line. Consequently, the difference
    # true_mse(l1) - true_mse(l2) ~= condensed_mse(l1) - condensed_mse(l2) for
    # any lines l1 and l2.
    self.assertAlmostEqual(l1_mse - l2_mse, condensed_l1_mse - condensed_l2_mse)

  def test_linear_condense_returns_xs_within_the_original_domain(self):
    np.random.seed(954350)
    x = np.random.uniform(low=0.0, high=1.0, size=100)
    y = x**2 + np.random.normal(scale=1.3, size=100)
    w = np.random.uniform(size=100)
    condensed_x, _, _ = linear_condense.linear_condense(x, y, w)
    self.assertLessEqual(x.min(), min(condensed_x))
    self.assertGreaterEqual(x.max(), max(condensed_x))

  def test_linear_condense_returns_positive_weights(self):
    np.random.seed(954351)
    x = np.random.uniform(low=0.0, high=1.0, size=100)
    y = x**2 + np.random.normal(scale=1.3, size=100)
    w = np.random.uniform(size=100)
    _, _, (w1, w2) = linear_condense.linear_condense(x, y, w)
    self.assertGreater(w1, 0)
    self.assertGreater(w2, 0)

  def test_condense_around_knots_does_nothing_if_already_condensed(self):
    np.random.seed(102034)
    x = np.sort(np.random.normal(size=1000))
    y = np.random.normal(size=1000)
    w = np.random.uniform(size=1000)
    knot_xs = np.sort(np.random.choice(x, size=37, replace=False))
    condensed_x, condensed_y, condensed_w = (
        linear_condense.condense_around_knots(x, y, w, knot_xs))
    np.testing.assert_array_equal(
        (condensed_x, condensed_y, condensed_w),
        linear_condense.condense_around_knots(condensed_x, condensed_y,
                                              condensed_w, knot_xs))

  def test_condense_around_knots_matches_mse_with_random_pwl_curves(self):
    # Given two PWLCurves on a set of knots, the condensed points around those
    # knots should preserve the MSE diff between those curves.
    np.random.seed(5)
    x = np.sort(np.random.normal(size=100))
    y = x + np.random.normal(size=100)
    w = np.random.uniform(size=100)

    # Select four random xs to serve as knots.
    knot_xs = np.sort(np.random.choice(x, size=4, replace=False))

    # Generate two random PWLCurves using the knot_xs.
    knot_ys_1 = np.random.normal(size=len(knot_xs))
    knot_ys_2 = np.random.normal(size=len(knot_xs))
    pwlcurve_1 = pwlcurve.PWLCurve(list(zip(knot_xs, knot_ys_1)))
    pwlcurve_2 = pwlcurve.PWLCurve(list(zip(knot_xs, knot_ys_2)))

    # Now generate the condensed points.
    condensed_x, condensed_y, condensed_w = (
        linear_condense.condense_around_knots(x, y, w, knot_xs))
    self.assertLessEqual(len(condensed_x), 2 * len(knot_xs) - 2)

    # Ensure delta line MSEs are the same on the full and condensed data.
    full_data_mse_1 = _curve_error_on_points(x, y, w, pwlcurve_1)
    full_data_mse_2 = _curve_error_on_points(x, y, w, pwlcurve_2)
    full_data_mse_delta = full_data_mse_1 - full_data_mse_2

    condensed_mse_1 = _curve_error_on_points(condensed_x, condensed_y,
                                             condensed_w, pwlcurve_1)
    condensed_mse_2 = _curve_error_on_points(condensed_x, condensed_y,
                                             condensed_w, pwlcurve_2)
    condensed_mse_delta = condensed_mse_1 - condensed_mse_2

    self.assertAlmostEqual(full_data_mse_delta, condensed_mse_delta)

  def test_condense_around_knots_with_repeated_points_on_knots(self):
    # For (x,y,w) such that x is in knot_xs, condense_around_knots must
    # decide whether to put (x,y,w) in the lower condensed range [prev_knot, x]
    # or the higher condensed range [x, next_knot]. Either way is fine, so long
    # as x is placed in precisely one of the two ranges. This test ensures that
    # condense_around_knots handles such (x,y,w) correctly.
    np.random.seed(12)
    knot_xs = np.array([1., 5., 8., 10.])
    x = np.sort(np.random.randint(10, size=1000))
    y = x + np.random.normal(size=len(x))
    w = np.random.uniform(size=len(x))

    # Generate two random PWLCurves using the knot_xs.
    knot_ys_1 = np.random.normal(size=len(knot_xs))
    knot_ys_2 = np.random.normal(size=len(knot_xs))
    pwlcurve_1 = pwlcurve.PWLCurve(list(zip(knot_xs, knot_ys_1)))
    pwlcurve_2 = pwlcurve.PWLCurve(list(zip(knot_xs, knot_ys_2)))

    # Now generate the condensed points.
    condensed_x, condensed_y, condensed_w = (
        linear_condense.condense_around_knots(x, y, w, knot_xs))
    self.assertLessEqual(len(condensed_x), 2 * len(knot_xs) - 2)

    # Ensure delta line MSEs are the same on the full and condensed data.
    full_data_mse_1 = _curve_error_on_points(x, y, w, pwlcurve_1)
    full_data_mse_2 = _curve_error_on_points(x, y, w, pwlcurve_2)
    full_data_mse_delta = full_data_mse_1 - full_data_mse_2

    condensed_mse_1 = _curve_error_on_points(condensed_x, condensed_y,
                                             condensed_w, pwlcurve_1)
    condensed_mse_2 = _curve_error_on_points(condensed_x, condensed_y,
                                             condensed_w, pwlcurve_2)
    condensed_mse_delta = condensed_mse_1 - condensed_mse_2

    self.assertAlmostEqual(full_data_mse_delta, condensed_mse_delta)


class SamplingTest(test_util.PWLFitTest):

  def test_sample_condense_points_invariant_to_y_order(self):
    # condense_around_knots requires points to be ordered by x, but doesn't care
    # whether the points are secondarily sorted by y. Among (x,y) pairs with the
    # same x, order doesn't matter.
    np.random.seed(5)
    x = np.sort(np.random.randint(0, 100, 1000).astype(float))
    y = np.random.uniform(0, 1, 1000)
    w = np.ones_like(x)

    # Randomize order of (x,y) pairs and then re-sort by x but not by y.
    xy = np.array([x, y]).T
    np.random.shuffle(xy)
    x_shuffled, y_shuffled = xy.T
    x_order = np.argsort(x_shuffled)
    x_shuffled, y_shuffled = x_shuffled[x_order], y_shuffled[x_order]

    self.assertTrue((x == x_shuffled).all())
    self.assertFalse((y == y_shuffled).all())
    expected_knots, expected_x, expected_y, expected_w = (
        linear_condense.sample_condense_points(x, y, w, 100))
    shuffled_knots, shuffled_x, shuffled_y, shuffled_w = (
        linear_condense.sample_condense_points(x_shuffled, y_shuffled, w, 100))

    # Knots should be identical, but the condensed x, y, and w might differ
    # slightly due to reordering floating point calculations.
    np.testing.assert_array_equal(expected_knots, shuffled_knots)
    self.assert_allclose(expected_x, shuffled_x)
    self.assert_allclose(expected_y, shuffled_y)
    self.assert_allclose(expected_w, shuffled_w)

  def test_sample_condense_points_invariant_to_fusion(self):
    # condense_around_knots doesn't care whether its x-values are unique,
    # or whether points with the same x-value have been fused ahead of time.
    np.random.seed(5)
    x = np.sort(np.random.randint(0, 100, size=777).astype(float))
    y = np.random.uniform(size=777)
    w = np.random.uniform(size=777)

    x_fused, y_fused, w_fused = utils.fuse_sorted_points(x, y, w)

    np.testing.assert_equal(
        linear_condense.sample_condense_points(x, y, w, 100),
        linear_condense.sample_condense_points(x_fused, y_fused, w_fused, 100))

  def test_sample_condense_points_when_num_knots_at_least_num_unique_xs(self):
    np.random.seed(5)
    # Generate x with more than 100 elements and exactly 100 unique elements.
    x = np.concatenate([np.arange(100), np.random.randint(100, size=100)])
    x = np.sort(x.astype(float))
    w = np.random.uniform(size=len(x))
    self.assertEqual(len(set(x)), 100)

    sampled_99 = linear_condense.sample_condense_points(x, x, w, 99)
    sampled_100 = linear_condense.sample_condense_points(x, x, w, 100)
    sampled_101 = linear_condense.sample_condense_points(x, x, w, 101)
    sampled_1000 = linear_condense.sample_condense_points(x, x, w, 1000)

    # There are 100 unique xs, so we need 100 samples to perfectly samples x.
    # Additional samples shouldn't matter.
    self.assertNotEqual(sampled_99, sampled_100)
    np.testing.assert_equal(sampled_100, sampled_101)
    np.testing.assert_equal(sampled_100, sampled_1000)

  def test_sample_condense_points_fused_data_when_three_unique_xs(self):
    x = np.array([0.0, 1.0, 2.0, 2.0, 2.0])
    y = x*x
    w = np.ones_like(x)
    expected_knots = np.array([0., 1., 2.])
    expected_x = np.array([0., 1., 2.])
    expected_y = np.array([0., 1., 4.])
    expected_w = np.array([1., 1., 3.])
    np.testing.assert_equal(
        [expected_knots, expected_x, expected_y, expected_w],
        linear_condense.sample_condense_points(x, y, w, 3))

  def test_sample_condense_points_fused_data_when_two_unique_xs(self):
    x = np.array([0.0, 0.0, 2.0, 2.0, 2.0])
    y = x*x
    w = np.array([1.0, 1.0, 3.0, 3.0, 3.0])
    expected_knots = np.array([0., 2.])
    expected_x = np.array([0., 2.])
    expected_y = np.array([0., 4.])
    expected_w = np.array([2., 9.])
    np.testing.assert_equal(
        [expected_knots, expected_x, expected_y, expected_w],
        linear_condense.sample_condense_points(x, y, w, 100))

  def test_sample_condense_points_fused_data_when_one_unique_x(self):
    x = np.array([1.0, 1.0, 1.0])
    y = np.array([2.0, 1.0, 3.0])
    w = np.ones_like(x)
    expected_knots = np.array([1.])
    expected_x = np.array([1.])
    expected_y = np.array([2.])
    expected_w = np.array([3.])
    np.testing.assert_equal(
        [expected_knots, expected_x, expected_y, expected_w],
        linear_condense.sample_condense_points(x, y, w, 100))

  def test_sample_condense_points_uses_two_condensed_points_per_knot_pair(self):
    np.random.seed(12)
    size = 1432
    x = np.sort(np.random.uniform(size=size))
    y = np.random.normal(size=size)
    w = np.random.uniform(size=size)
    knot_xs, condensed_x, condensed_y, condensed_w = (
        linear_condense.sample_condense_points(x, y, w, 100))

    # Two condensed points between each pair --> 2 * (#knots - 1) total.
    self.assertEqual(len(knot_xs), 100)
    self.assertEqual(len(condensed_y), 198)
    self.assertEqual(len(condensed_w), 198)
    self.assertEqual(len(condensed_x), 198)

  def test_condense_around_knots_matches_mse_with_random_pwl_curves(self):
    # condense_around_knots picks a set of candidate knots and then
    # linearly condenses points around those candidate knots. Given any two
    # piecewise-linear curves defined on candidate knots, the condensed points
    # should preserve the MSE diff between those curves.
    np.random.seed(13)
    x = np.sort(np.random.normal(size=937))
    y = x + np.random.normal(size=len(x))
    w = np.random.uniform(size=len(x))

    knot_xs, condensed_x, condensed_y, condensed_w = (
        linear_condense.sample_condense_points(x, y, w, 100))

    # Generate two random PWLCurves using the knot_xs.
    knot_ys_1 = np.random.normal(size=len(knot_xs))
    knot_ys_2 = np.random.normal(size=len(knot_xs))
    pwlcurve_1 = pwlcurve.PWLCurve(list(zip(knot_xs, knot_ys_1)))
    pwlcurve_2 = pwlcurve.PWLCurve(list(zip(knot_xs, knot_ys_2)))

    # Ensure delta line MSEs are the same on the full and condensed data.
    full_data_mse_1 = _curve_error_on_points(x, y, w, pwlcurve_1)
    full_data_mse_2 = _curve_error_on_points(x, y, w, pwlcurve_2)
    full_data_mse_delta = full_data_mse_1 - full_data_mse_2

    condensed_mse_1 = _curve_error_on_points(condensed_x, condensed_y,
                                             condensed_w, pwlcurve_1)
    condensed_mse_2 = _curve_error_on_points(condensed_x, condensed_y,
                                             condensed_w, pwlcurve_2)
    condensed_mse_delta = condensed_mse_1 - condensed_mse_2

    self.assertAlmostEqual(full_data_mse_delta, condensed_mse_delta)


if __name__ == '__main__':
  unittest.main()
