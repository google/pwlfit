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


"""Tests for pwlfit/fitter."""

import unittest

import numpy as np
from pwlfit import fitter
from pwlfit import pwlcurve
from pwlfit import test_util
from pwlfit import transform


def pwl_predict(x, y, *args, **kwargs):
  """Test utility to fit and evaluate a curve in one function."""
  return fitter.fit_pwl(x, y, *args, **kwargs).eval(x)


def _curve_error_on_points(x, y, w, curve):
  """Squared error when using the curve to predict the points."""
  return np.sum((y - curve.eval(x))**2 * w)


def count_slope_inversions(ys):
  """Count how many times the slope changes sign."""
  num_inversions = 0
  for a, b, c in zip(ys[:-2], ys[1:-1], ys[2:]):
    if (b - a) * (c - b) < 0:
      num_inversions += 1
  return num_inversions


class FitterTest(test_util.PWLFitTest):

  def test_mono_type_enum_and_bool(self):
    np.random.seed(48440)
    x = np.sort(np.random.uniform(size=100))
    y = np.random.normal(size=100)
    w = np.random.uniform(size=100)
    # fit_pwl treats mono=True as synonymous to mono=MonoType.mono.
    self.assertEqual(fitter.fit_pwl(x, y, w, 2, mono=True),
                     fitter.fit_pwl(x, y, w, 2, mono=fitter.MonoType.mono))
    self.assertNotEqual(
        fitter.fit_pwl(x, y, w, 2, mono=True),
        fitter.fit_pwl(x, y, w, 2, mono=fitter.MonoType.nonmono))

    # fit_pwl treats mono=False as synonymous to mono=MonoType.nonmono.
    self.assertEqual(fitter.fit_pwl(x, y, w, 2, mono=False),
                     fitter.fit_pwl(x, y, w, 2, mono=fitter.MonoType.nonmono))
    self.assertNotEqual(fitter.fit_pwl(x, y, w, 2, mono=False),
                        fitter.fit_pwl(x, y, w, 2, mono=fitter.MonoType.mono))

  def test_mono_increasing_line(self):
    x = np.arange(51, dtype=float)
    y = pwlcurve.PWLCurve([(0, -1), (50, 75)]).eval(x)
    w = np.ones_like(x)
    self.assert_allclose(y, pwl_predict(x, y, w, 1))
    self.assert_allclose(y, pwl_predict(x, y, w, 2))

  def test_mono_decreasing_line(self):
    x = np.arange(51, dtype=float)
    y = pwlcurve.PWLCurve([(0, 75), (50, -1)]).eval(x)
    w = np.ones_like(x)
    self.assert_allclose(y, pwl_predict(x, y, w, 1))
    self.assert_allclose(y, pwl_predict(x, y, w, 2))

  def test_mono_decreasing_log1p_line(self):
    x = np.arange(51, dtype=float)
    y = pwlcurve.PWLCurve([(x[0], 75), (x[-1], -1)], np.log1p).eval(x)
    w = np.ones_like(x)
    self.assert_allclose(y, pwl_predict(x, y, w, 1))
    self.assert_allclose(y, pwl_predict(x, y, w, 2))

  def test_simple_slope_restrictions(self):
    x = np.arange(51, dtype=float)
    y = pwlcurve.PWLCurve([(0, 0), (50, 75)]).eval(x)
    w = np.ones_like(x)

    # The true slope is 1.5, which is compatible with any slope restrictions
    # that allow slope=1.5.
    self.assert_allclose(y, pwl_predict(x, y, w, 1, min_slope=1))
    self.assert_allclose(y, pwl_predict(x, y, w, 1, max_slope=2))
    self.assert_allclose(y, pwl_predict(x, y, w, 1, min_slope=1, mono=False))
    self.assert_allclose(y, pwl_predict(x, y, w, 1, max_slope=2, mono=False))

    # An ideal fit isn't possible if we prevent slope=1.5.
    self.assert_notallclose(y, pwl_predict(x, y, w, 1, max_slope=1))
    self.assert_notallclose(y, pwl_predict(x, y, w, 1, min_slope=2))
    self.assert_notallclose(y, pwl_predict(x, y, w, 1, max_slope=1, mono=False))
    self.assert_notallclose(y, pwl_predict(x, y, w, 1, min_slope=2, mono=False))

  def test_learn_ends_has_no_effect_when_endpoints_are_ideal(self):
    # In this case, the ideal fit uses the endpoints, so no need to learn ends.
    x = np.arange(51, dtype=float)
    y = pwlcurve.PWLCurve([(0, 0), (10, 1), (25, 25), (50, 60)]).eval(x)
    w = np.ones_like(x)

    self.assertEqual(fitter.fit_pwl(x, y, w, 3, learn_ends=True),
                     fitter.fit_pwl(x, y, w, 3, learn_ends=False))

  def test_one_segment_pwl_with_flat_ends(self):
    x = np.arange(51, dtype=float)
    y = pwlcurve.PWLCurve([(0, 0), (10, 0), (40, 50), (50, 50)]).eval(x)
    w = np.ones_like(x)
    # A one-segment PWLCurve can fit fn perfectly, but only if its knots are
    # [(10, 0), (40, 50)]. This test confirms that fit_pwl learns those knots.
    curve = fitter.fit_pwl(x, y, w, 1)
    self.assert_allclose([(10, 0), (40, 50)], curve.points)
    self.assert_allclose(y, curve.eval(x))
    self.assertEqual(transform.identity, curve.fx)

  def test_one_segment_pwl_with_flat_ends_but_no_learning_ends(self):
    x = np.arange(51, dtype=float)
    y = pwlcurve.PWLCurve([(0, 0), (10, 0), (40, 50), (50, 50)]).eval(x)
    w = np.ones_like(x)
    # A one-segment PWLCurve can fit fn perfectly, but only if its knots are
    # [(10, 0), (40, 50)]. In this test, we disable learn_ends, and show that
    # the fitter can't learn the ideal fit because it's forced to use 0 and 50
    # as control points.
    curve = fitter.fit_pwl(x, y, w, 1, learn_ends=False)
    self.assertEqual([0, 50], curve.xs)
    self.assert_notallclose(y, curve.eval(x))

  def test_mono_increasing_two_segment_pwl(self):
    x = np.arange(51, dtype=float)
    y = pwlcurve.PWLCurve([(0, 0), (25, 25), (50, 60)]).eval(x)
    w = np.ones_like(x)

    self.assert_allclose(y, pwl_predict(x, y, w, 2))
    self.assert_allclose(y, pwl_predict(x, y, w, 3))
    self.assert_allclose(y, pwl_predict(x, y, w, 4))

  def test_mono_decreasing_two_segment_pwl(self):
    x = np.arange(51, dtype=float)
    y = pwlcurve.PWLCurve([(0, 60), (25, 25), (50, 0)]).eval(x)
    w = np.ones_like(x)

    self.assert_allclose(y, pwl_predict(x, y, w, 2))
    self.assert_allclose(y, pwl_predict(x, y, w, 3))
    self.assert_allclose(y, pwl_predict(x, y, w, 4))

  def test_non_mono_two_segment_pwl(self):
    x = np.arange(51, dtype=float)
    y = pwlcurve.PWLCurve([(0, 0), (25, 25), (50, 0)]).eval(x)
    w = np.ones_like(x)

    # Unfortunately, fitter will learn a log1p transform for this problem unless
    # we override transforms.
    self.assert_allclose(
        y, pwl_predict(x, y, w, 2, mono=False, fx=transform.identity))
    self.assert_allclose(
        y, pwl_predict(x, y, w, 3, mono=False, fx=transform.identity))
    self.assert_allclose(
        y, pwl_predict(x, y, w, 4, mono=False, fx=transform.identity))

    # Monotone curves can't fit this data closely.
    self.assert_notallclose(
        y, pwl_predict(x, y, w, 2, mono=True, fx=transform.identity))

  def test_non_mono_two_segment_log(self):
    exp_x = 1 + np.arange(51, dtype=float)
    x = np.log(exp_x)
    y = pwlcurve.PWLCurve([(x[0], 0), (x[25], 25), (x[50], 0)]).eval(x)
    w = np.ones_like(x)

    # Piecewise-linear in log space.
    self.assert_allclose(y, pwl_predict(exp_x, y, w, 2, mono=False, fx=np.log))
    self.assert_allclose(y, pwl_predict(exp_x, y, w, 3, mono=False, fx=np.log))
    self.assert_allclose(y, pwl_predict(exp_x, y, w, 4, mono=False, fx=np.log))

    # Monotone curves can't fit this data closely.
    self.assert_notallclose(y,
                            pwl_predict(exp_x, y, w, 2, mono=True, fx=np.log))

  def test_mono_increasing_two_segment_pwl_with_flat_ends(self):
    x = np.arange(51, dtype=float)
    y = pwlcurve.PWLCurve([(0, 0), (10, 0), (25, 15), (40, 50),
                           (50, 50)]).eval(x)
    w = np.ones_like(x)
    # A two-segment PWLCurve can fit fn perfectly, but only if its knots are
    # [(10, 0), (25, 15), (40, 50)]. This test confirms that fit_pwl will
    # learn those knots.
    curve = fitter.fit_pwl(x, y, w, 2)
    self.assert_allclose([(10, 0), (25, 15), (40, 50)], curve.points)
    self.assert_allclose(y, curve.eval(x))
    self.assertEqual(transform.identity, curve.fx)

  def test_non_mono_two_segment_pwl_with_flat_ends(self):
    x = np.arange(51, dtype=float)
    y = pwlcurve.PWLCurve([(0, 0), (10, 0), (25, 15), (40, 0), (50, 0)]).eval(x)
    w = np.ones_like(x)
    # A two-segment PWLCurve can fit fn perfectly, but only if its knots are
    # [(10, 0), (25, 15), (40, 0)]. This test confirms that fit_pwl will learn
    # those knots.
    curve = fitter.fit_pwl(x, y, w, 2, mono=False, fx=transform.identity)
    self.assert_allclose([(10, 0), (25, 15), (40, 0)], curve.points)
    self.assert_allclose(y, curve.eval(x))

  def test_non_mono_two_segment_pwl_with_flat_ends_but_no_learning_ends(self):
    x = np.arange(51, dtype=float)
    y = pwlcurve.PWLCurve([(0, 0), (10, 0), (25, 15), (40, 0), (50, 0)]).eval(x)
    w = np.ones_like(x)
    # A two-segment PWLCurve can fit fn perfectly, but only if its knots are
    # [(10, 0), (25, 15), (40, 0)]. In this test, we disable learn_ends, and
    # show that the fitter can't learn the ideal fit because it's forced to use
    # 0 and 50 as control points.
    curve = fitter.fit_pwl(
        x, y, w, 2, mono=False, fx=transform.identity, learn_ends=False)
    self.assertEqual([0, 25, 50], curve.xs)
    self.assert_notallclose(y, curve.eval(x))

  def test_mono_increasing_three_segment_pwl(self):
    x = np.arange(51, dtype=float)
    y = pwlcurve.PWLCurve([(0, 0), (10, 1), (25, 25), (50, 60)]).eval(x)
    w = np.ones_like(x)

    self.assert_allclose(y, pwl_predict(x, y, w, 3))
    self.assert_allclose(y, pwl_predict(x, y, w, 4))

  def test_mono_decreasing_three_segment_pwl(self):
    x = np.arange(51, dtype=float)
    y = pwlcurve.PWLCurve([(0, 60), (10, 25), (25, 1), (50, 0)]).eval(x)
    w = np.ones_like(x)

    self.assert_allclose(
        y, pwl_predict(x, y, w, 3, mono=False, fx=transform.identity))
    self.assert_allclose(
        y, pwl_predict(x, y, w, 4, mono=False, fx=transform.identity))

  def test_non_mono_three_segment_pwl(self):
    x = np.arange(51, dtype=float)
    y = pwlcurve.PWLCurve([(0, 0), (10, 25), (25, 10), (50, 60)]).eval(x)
    w = np.ones_like(x)

    self.assert_allclose(
        y, pwl_predict(x, y, w, 3, mono=False, fx=transform.identity))
    self.assert_allclose(
        y, pwl_predict(x, y, w, 4, mono=False, fx=transform.identity))

  def test_mono_increasing_four_segment_pwl(self):
    x = np.arange(51, dtype=float)
    y = pwlcurve.PWLCurve([(0, 0), (10, 1), (25, 25), (30, 59),
                           (50, 60)]).eval(x)
    w = np.ones_like(x)

    self.assert_allclose(y, pwl_predict(x, y, w, 4))
    self.assert_allclose(y, pwl_predict(x, y, w, 5))

  def test_mono_decreasing_four_segment_pwl(self):
    x = np.arange(51, dtype=float)
    y = pwlcurve.PWLCurve([(0, 60), (10, 59), (25, 25), (30, 1),
                           (50, 0)]).eval(x)
    w = np.ones_like(x)

    self.assert_allclose(y, pwl_predict(x, y, w, 4))
    self.assert_allclose(y, pwl_predict(x, y, w, 5))

  def test_non_mono_four_segment_pwl(self):
    x = np.arange(51, dtype=float)
    y = pwlcurve.PWLCurve([(0, 0), (10, 25), (26, 10), (35, 59),
                           (50, 5)]).eval(x)
    w = np.ones_like(x)

    self.assert_allclose(
        y, pwl_predict(x, y, w, 4, mono=False, fx=transform.identity))
    self.assert_allclose(
        y, pwl_predict(x, y, w, 5, mono=False, fx=transform.identity))

  def test_fit_pwl_with_weights(self):
    x = np.array([0., 25., 50])
    y = pwlcurve.PWLCurve([(0, 0), (25, 30), (50, 50)]).eval(x)
    w = np.array([1., 1., 2.])

    # The fit with weights tip up on low end but down on the high end.
    pred_ys_weightless = pwl_predict(x, y, np.ones_like(x), 1)
    pred_ys = pwl_predict(x, y, w, 1)
    self.assertLess(pred_ys_weightless[0], pred_ys[0])
    self.assertGreater(pred_ys_weightless[-1], pred_ys[-1])

    # With segments=2, weights have no effect since we can fit perfectly.
    self.assert_allclose(y, pwl_predict(x, y, np.ones_like(x), 2))
    self.assert_allclose(y, pwl_predict(x, y, w, 2))

  def test_fit_pwl_with_one_unique_x(self):
    x = np.ones(10, dtype=float)
    y = x * 5
    w = np.ones_like(x)
    self.assert_allclose(y, pwl_predict(x, y, w, 1))

  def test_fit_pwl_with_four_segment_unimodal(self):
    x = np.arange(51, dtype=float) / 10
    y = pwlcurve.PWLCurve([(0, 0), (1, 2), (2, 5), (3, 2), (4, 0)]).eval(x)

    self.assert_allclose(y, pwl_predict(
        x, y, num_segments=4, fx=transform.identity,
        mono=fitter.MonoType.bitonic))

  def test_fit_pwl_with_four_segment_unimodal_with_slope_restrictions(self):
    x = np.arange(51, dtype=float) / 10
    y = pwlcurve.PWLCurve([(0, 0), (1, 2), (2, 5), (3, 2), (4, 0)]).eval(x)

    # -3 <= true_slope <= 3. FitUnimodalPWL can find the ideal fit unless we
    # require a min_slope > -3 or a max_slope < 3.
    self.assert_allclose(y, pwl_predict(
        x, y, num_segments=4, fx=transform.identity,
        mono=fitter.MonoType.bitonic, min_slope=-3, max_slope=3))

    # Min slope is too large for a perfect fit.
    self.assert_notallclose(y, pwl_predict(
        x, y, num_segments=4, fx=transform.identity,
        mono=fitter.MonoType.bitonic, min_slope=-2))
    # Max slope is too small for a perfect fit.
    self.assert_notallclose(y, pwl_predict(
        x, y, num_segments=4, fx=transform.identity,
        mono=fitter.MonoType.bitonic, max_slope=2))

  def test_fit_pwl_unimodal_on_non_unimodal_data(self):
    x = np.arange(51, dtype=float) / 10
    y = pwlcurve.PWLCurve([(0, 0), (1, 2), (2, 0), (3, 2), (4, 0)]).eval(x)

    curve = fitter.fit_pwl(x, y, num_segments=4, fx=transform.identity,
                           mono=fitter.MonoType.bitonic)

    # An unrestricted fit should change directions three times, but a unimodal
    # curve will only change once.
    self.assertEqual(1, count_slope_inversions(curve.ys))

    # Should be concave down -- increasing at first, decreasing at the end.
    self.assertLess(curve.ys[0], curve.ys[1])
    self.assertLess(curve.ys[-1], curve.ys[-2])

  def test_fit_pwl_unimodal_with_forced_direction(self):
    x = np.arange(51, dtype=float) / 10
    y = pwlcurve.PWLCurve([(0, 0), (1, 2), (2, 5), (3, 2), (4, 0)]).eval(x)

    # y is concave down, so a concave solution is ideal.
    concave_curve = fitter.fit_pwl(x, y, num_segments=4, fx=transform.identity,
                                   mono=fitter.MonoType.bitonic_concave_down)
    convex_curve = fitter.fit_pwl(x, y, num_segments=4, fx=transform.identity,
                                  mono=fitter.MonoType.bitonic_concave_up)

    self.assert_allclose(y, concave_curve.eval(x))
    self.assert_notallclose(y, convex_curve.eval(x))

    # -y is concave up, so a convex solution is ideal.
    concave_curve = fitter.fit_pwl(x, -y, num_segments=4, fx=transform.identity,
                                   mono=fitter.MonoType.bitonic_concave_down)
    convex_curve = fitter.fit_pwl(x, -y, num_segments=4, fx=transform.identity,
                                  mono=fitter.MonoType.bitonic_concave_up)

    self.assert_allclose(-y, convex_curve.eval(x))
    self.assert_notallclose(-y, concave_curve.eval(x))


class FitPWLPointsTest(test_util.PWLFitTest):

  def test_fit_pwl_points_ignores_extra_segments(self):
    # A curve can optimally fit the sample with one knot per sample point. Any
    # additional knots would be wasted, so fit_pwl_points shouldn't generate
    # them.
    np.random.seed(48440)
    x = np.sort(np.random.uniform(size=3))
    y = np.random.normal(size=3)
    w = np.random.uniform(size=3)
    np.testing.assert_array_equal(
        fitter.fit_pwl_points(x, x, y, w, num_segments=2),
        fitter.fit_pwl_points(x, x, y, w, num_segments=3))

  def test_fit_pwl_points_returns_mean_when_one_knot(self):
    # A proper PWLCurve requires two knots. When given only one knot candidate,
    # fit_pwl_points should return a constant mean(y) PWLCurve.
    np.random.seed(48440)
    x_knots = np.array([.4])
    x = np.sort(np.random.uniform(size=100))
    y = x**2 + np.random.normal(scale=.1, size=100)
    w = np.random.uniform(size=100)
    _, y_points = fitter.fit_pwl_points(x_knots, x, y, w, num_segments=2)
    y_mean = np.average(y, weights=w)
    # y_points should all be y.mean().
    np.testing.assert_array_equal(y_mean * np.ones_like(y_points), y_points)

  def test_fit_pwl_points_mono_with_extra_segments(self):
    # Monotonicity constraints should hold even when fit_pwl_points is asked for
    # more segments than necessary.
    np.random.seed(48440)
    x = np.sort(np.random.uniform(size=3))
    y = -np.sort(np.random.normal(size=3))
    w = np.random.uniform(size=3)
    # Fit mono-decreasing data with mono-increasing constraints. That way, we'll
    # know if fit_pwl_points ignores the constraints.
    _, y_pnts = fitter.fit_pwl_points(x, x, y, w, num_segments=3, min_slope=0)
    self.assert_increasing(y_pnts)

  def test_fit_pwl_points_raises_when_min_slope_exceeds_max_slope(self):
    np.random.seed(48440)
    x = np.sort(np.random.uniform(size=3))
    y = -np.sort(np.random.normal(size=3))
    w = np.random.uniform(size=3)
    with self.assertRaises(ValueError):
      fitter.fit_pwl_points(x, x, y, w, 3, min_slope=1, max_slope=-1)

  def test_fit_pwl_points_required_x_knots_appear_in_solution(self):
    np.random.seed(48440)
    x = np.sort(np.random.uniform(size=100))
    y = -np.sort(np.random.normal(size=100))
    w = np.random.uniform(size=100)

    for num_required_x_knots in range(5):
      required_x_knots = np.random.uniform(size=num_required_x_knots)
      x_pnts, _ = fitter.fit_pwl_points(
          x, x, y, w, 3, required_x_knots=required_x_knots)
      self.assertContainsSubset(required_x_knots, x_pnts)

  def test_fit_pwl_points_required_x_knots_handles_empty_cases(self):
    np.random.seed(48440)
    x = np.sort(np.random.uniform(size=100))
    y = -np.sort(np.random.normal(size=100))
    w = np.random.uniform(size=100)

    self.assertEqual(
        fitter.fit_pwl_points(x, x, y, w, 3),
        fitter.fit_pwl_points(x, x, y, w, 3, required_x_knots=None))
    self.assertEqual(
        fitter.fit_pwl_points(x, x, y, w, 3),
        fitter.fit_pwl_points(x, x, y, w, 3, required_x_knots=[]))
    self.assertEqual(
        fitter.fit_pwl_points(x, x, y, w, 3),
        fitter.fit_pwl_points(x, x, y, w, 3, required_x_knots=np.array([])))

  def test_fit_pwl_points_raises_when_requiring_too_many_x_knots(self):
    np.random.seed(48440)
    x = np.sort(np.random.uniform(size=100))
    y = -np.sort(np.random.normal(size=100))
    w = np.random.uniform(size=100)

    # For 3 segments, we can require up to 4 knots.
    fitter.fit_pwl_points(x, x, y, w, 3, required_x_knots=x[[0, 1, 2, 3]])
    with self.assertRaises(ValueError):
      fitter.fit_pwl_points(x, x, y, w, 3, required_x_knots=x[[0, 1, 2, 3, 4]])

  def test_fit_pwl_points_required_x_knots(self):
    x = np.arange(51, dtype=float)
    y = pwlcurve.PWLCurve([(0, 0), (25, 25), (50, 0)]).eval(x)
    w = np.ones_like(x)

    # Optimal knots don't change the fit.
    self.assertEqual(
        fitter.fit_pwl_points(x, x, y, w, 2, required_x_knots=[0, 25]),
        fitter.fit_pwl_points(x, x, y, w, 2))

    # Suboptimal knots do change the fit.
    self.assert_notallclose(
        fitter.fit_pwl_points(x, x, y, w, 2, required_x_knots=[5]),
        fitter.fit_pwl_points(x, x, y, w, 2))

  def test_fit_pwl_points_non_mono_two_segment(self):
    x = np.arange(51, dtype=float)
    y = pwlcurve.PWLCurve([(0, 0), (25, 25), (50, 0)]).eval(x)
    w = np.ones_like(x)
    curve_xs, curve_ys = fitter.fit_pwl_points(x, x, y, w, 2)
    self.assert_allclose(curve_xs, [0, 25, 50])
    self.assert_allclose(curve_ys, [0, 25, 0])

  def test_solver_works_without_bounding_knots(self):
    x = np.array([0., 1., 2., 3., 4., 5., 6., 7.])
    y = np.array([0., 0., 0., 1., 2., 3., 4., 4.])
    w = np.ones_like(x)
    knots = [2., 3., 5., 6.]

    # This problem has a perfect monotone increasing solution, which the solver
    # should find whether it's set to mono-up or non-mono.
    solver = fitter._WeightedLeastSquaresPWLSolver(x, y, w)
    solution, error = solver.solve(knots)
    self.assert_allclose(solution, [0, 1, 3, 4])
    self.assertAlmostEqual(error, 0)

    solver = fitter._WeightedLeastSquaresPWLSolver(x, y, w, min_slope=0)
    solution, error = solver.solve(knots)
    self.assert_allclose(solution, [0, 1, 3, 4])
    self.assertAlmostEqual(error, 0)

  def test_knots_between_xs(self):
    x = np.array([0., 1., 2., 3., 4., 5., 6., 7.])
    y = np.array([0., 0., 0., 1., 2., 3., 4., 4.])
    w = np.ones_like(x)
    knots = [2.5, 3.5, 4.5, 5.5]

    # This problem has a perfect monotone increasing solution, which the solver
    # should find whether it's set to mono-up or non-mono.
    solver = fitter._WeightedLeastSquaresPWLSolver(x, y, w)
    solution, residue = solver.solve(knots)
    self.assert_allclose(solution, [0, 2, 2, 4])
    self.assertAlmostEqual(residue, 0)

    solver = fitter._WeightedLeastSquaresPWLSolver(x, y, w, min_slope=0)
    solution, residue = solver.solve(knots)
    self.assert_allclose(list(solution), [0, 2, 2, 4])
    self.assertAlmostEqual(residue, 0)

  def test_reports_correct_error(self):
    np.random.seed(58440)
    x = np.sort(np.random.uniform(size=100))
    y = x**2 + np.random.normal(scale=.2, size=100)
    w = np.random.uniform(size=100)
    knots = [.2, .5, .8, .9]

    solver = fitter._WeightedLeastSquaresPWLSolver(x, y, w)
    knot_ys, reported_error = solver.solve(knots)
    points = list(zip(knots, knot_ys))
    true_error = _curve_error_on_points(x, y, w, pwlcurve.PWLCurve(points))
    self.assertAlmostEqual(true_error, reported_error)

    mono_solver = fitter._WeightedLeastSquaresPWLSolver(x, y, w, min_slope=0)
    knot_ys, reported_error = mono_solver.solve(knots)
    points = list(zip(knots, knot_ys))
    true_error = _curve_error_on_points(x, y, w, pwlcurve.PWLCurve(points))
    self.assertAlmostEqual(true_error, reported_error)

  def test_mono_down_equals_flipped_mono_up(self):
    # Fitting monoup should be equivalent to flipping y and fitting monodown.
    np.random.seed(58442)
    x = np.sort(np.random.uniform(size=100))
    y = x**2 + np.random.normal(scale=.2, size=100)
    w = np.random.uniform(size=100)
    mono_up_solver = fitter._WeightedLeastSquaresPWLSolver(x, y, w, min_slope=0)
    mono_down_solver = fitter._WeightedLeastSquaresPWLSolver(
        x, -y, w, max_slope=0)
    knots = [.2, .5, .8, .9]
    mono_up_y, mono_up_error = mono_up_solver.solve(knots)
    mono_down_y, mono_down_error = mono_down_solver.solve(knots)
    self.assertAlmostEqual(mono_up_error, mono_down_error)
    self.assert_allclose(mono_up_y, -mono_down_y)

  def test_mono_solver_returns_mono_solution(self):
    np.random.seed(58443)
    x = np.sort(np.random.uniform(size=10))
    y = np.random.normal(scale=.5, size=10)
    w = np.random.uniform(size=10)
    knots = np.sort(np.random.choice(x, size=4, replace=False))
    mono_up_solver = fitter._WeightedLeastSquaresPWLSolver(x, y, w, min_slope=0)
    mono_down_solver = fitter._WeightedLeastSquaresPWLSolver(
        x, y, w, max_slope=0)

    mono_up_y, _ = mono_up_solver.solve(knots)
    self.assert_increasing(mono_up_y)
    mono_down_y, _ = mono_down_solver.solve(knots)
    self.assert_decreasing(mono_down_y)

  def test_sloped_solver_returns_sloped_solution(self):
    np.random.seed(58443)
    x = np.sort(np.random.uniform(size=10))
    y = np.random.normal(scale=.5, size=10)
    w = np.random.uniform(size=10)
    knots = np.sort(np.random.choice(x, size=4, replace=False))
    min_slope_solver = fitter._WeightedLeastSquaresPWLSolver(
        x, y, w, min_slope=5.)
    max_slope_solver = fitter._WeightedLeastSquaresPWLSolver(
        x, y, w, max_slope=-5.)

    min_slope_y, _ = min_slope_solver.solve(knots)
    slopes = (min_slope_y[1:] - min_slope_y[:1]) / (knots[1:] - knots[:-1])
    for slope in slopes:
      self.assertGreaterEqual(slope, 5)

    max_slope_y, _ = max_slope_solver.solve(knots)
    slopes = (max_slope_y[1:] - max_slope_y[:1]) / (knots[1:] - knots[:-1])
    for slope in slopes:
      self.assertLessEqual(slope, -5)

  def test_max_slope_smooths_out_spike(self):
    x = np.array([0., 2., 4., 6., 8., 10.])
    y = np.array([0., 0., 0., 6., 6., 6.])
    w = np.ones_like(x)
    knots = x
    max_slope_solver = fitter._WeightedLeastSquaresPWLSolver(
        x, y, w, max_slope=1.0)
    max_slope_y, _ = max_slope_solver.solve(knots)
    self.assert_allclose(max_slope_y, [0, 0, 2, 4, 6, 6])

  def test_min_and_max_slope(self):
    x = np.array([0., 1., 2., 3., 4.])
    y = np.array([0., 2., 0., -2, 0.])
    w = np.ones_like(x)
    knots = x
    slope_solver = fitter._WeightedLeastSquaresPWLSolver(
        x, y, w, min_slope=-1.0, max_slope=1.0)
    knot_ys, _ = slope_solver.solve(knots)
    self.assert_allclose(knot_ys, [0, 1, 0, -1, 0])

  def test_mono_solver_same_as_non_mono_for_mono_task(self):
    # The monotonicity restriction shouldn't matter if the natural fit is
    # monotonic.
    np.random.seed(58444)
    x = np.sort(np.random.uniform(size=100))
    y = x**2
    w = np.random.uniform(size=100)
    knots = [.2, .5, .8, .9]

    mono_up_solver = fitter._WeightedLeastSquaresPWLSolver(x, y, w, min_slope=0)
    non_mono_solver = fitter._WeightedLeastSquaresPWLSolver(x, y, w)
    mono_up_y, mono_up_error = mono_up_solver.solve(knots)
    non_mono_y, non_mono_error = non_mono_solver.solve(knots)
    np.testing.assert_array_equal(non_mono_y, mono_up_y)
    self.assertAlmostEqual(mono_up_error, non_mono_error)

  def test_bitonic_solver_same_as_non_mono_for_bitonic_task(self):
    # The bitonic restriction shouldn't matter if the natural fit is bitonic.
    np.random.seed(58444)
    x = np.sort(np.random.uniform(size=100))
    y = (0.6 - x)**2  # y is a concave-up parabole with its minimum at x=0.6.
    w = np.random.uniform(size=100)
    knots = [.2, .5, .7, .9]

    concave_up_solver = fitter._WeightedLeastSquaresPWLSolver(
        x, y, w, bitonic_peak=0.6, bitonic_concave_down=False)
    non_mono_solver = fitter._WeightedLeastSquaresPWLSolver(x, y, w)

    concave_up_y, concave_up_error = concave_up_solver.solve(knots)
    non_mono_y, non_mono_error = non_mono_solver.solve(knots)

    # The problem is bitonic concave up, so the concave up solution matches the
    # non-mono.
    np.testing.assert_array_equal(non_mono_y, concave_up_y)
    self.assertAlmostEqual(concave_up_error, non_mono_error)

  def test_bitonic_solver_suffers_when_peak_or_direction_is_wrong(self):
    np.random.seed(58444)
    x = np.sort(np.random.uniform(size=100))
    y = (0.5 - x)**2  # y is a concave-up parabole with its minimum at x=0.6.
    w = np.random.uniform(size=100)
    knots = [.1, .3, .5, .7, .9]

    non_mono_solver = fitter._WeightedLeastSquaresPWLSolver(x, y, w)
    concave_down_solver = fitter._WeightedLeastSquaresPWLSolver(
        x, y, w, bitonic_peak=0.5, bitonic_concave_down=True)
    peak_too_low_solver = fitter._WeightedLeastSquaresPWLSolver(
        x, y, w, bitonic_peak=0.2, bitonic_concave_down=False)
    peak_too_high_solver = fitter._WeightedLeastSquaresPWLSolver(
        x, y, w, bitonic_peak=0.8, bitonic_concave_down=False)

    non_mono_y, non_mono_error = non_mono_solver.solve(knots)
    concave_down_y, concave_down_error = concave_down_solver.solve(knots)
    peak_too_low_y, peak_too_low_error = peak_too_low_solver.solve(knots)
    peak_too_high_y, peak_too_high_error = peak_too_high_solver.solve(knots)

    # The bitonic restrictions prevent us from learning the good solution.
    self.assert_notallclose(non_mono_y, concave_down_y)
    self.assert_notallclose(non_mono_y, peak_too_low_y)
    self.assert_notallclose(non_mono_y, peak_too_high_y)

    self.assertNotAlmostEqual(concave_down_error, non_mono_error)
    self.assertNotAlmostEqual(peak_too_low_error, non_mono_error)
    self.assertNotAlmostEqual(peak_too_high_error, non_mono_error)

  def test_solver_squared_error_in_the_underdetermined_case(self):
    x = np.array([0., 4., 5.])
    y = np.array([0., 0., 2.])
    w = np.ones_like(x)
    knots = np.array([1., 2., 3.])

    # The x=1 and x=3 knots affect the squared error, so their y-values are
    # determined by the system. However, the x=2 knot has no effect, so it can
    # take on an infinite range of values.
    # The expected curve is [(1., 0), (2., unknown), (3., 1.)].
    expected_error = 2.0

    nonmono_solver = fitter._WeightedLeastSquaresPWLSolver(x, y, w)
    _, nonmono_error = nonmono_solver.solve(knots)
    self.assertAlmostEqual(expected_error, nonmono_error)

    mono_up_solver = fitter._WeightedLeastSquaresPWLSolver(x, y, w, min_slope=0)
    _, mono_up_error = mono_up_solver.solve(knots)
    self.assertAlmostEqual(expected_error, mono_up_error)

    # The mono-down case is fully constrained, with best curve y = mean = 2/3.
    mono_down_solver = fitter._WeightedLeastSquaresPWLSolver(
        x, y, w, max_slope=0)
    _, mono_down_error = mono_down_solver.solve(knots)
    self.assertAlmostEqual(8./3, mono_down_error)

if __name__ == '__main__':
  unittest.main()
