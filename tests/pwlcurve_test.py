# Copyright 2020 Google LLC
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
"""Tests for pwlfit/curve."""
import unittest
from absl.testing import parameterized

import numpy as np
from pwlfit import pwlcurve
from pwlfit import test_util
from pwlfit import transform


class PWLCurveTest(test_util.PWLFitTest, parameterized.TestCase):

  def test_simple(self):
    curve = pwlcurve.PWLCurve([(1, 10), (2, 20)])
    self.assertEqual([(1, 10), (2, 20)], curve.curve_points)
    self.assertEqual([1, 2], curve.curve_xs)
    self.assertEqual([10, 20], curve.curve_ys)
    self.assertIs(transform.identity, curve.xform)

  def test_not_enough_points(self):
    with self.assertRaises(ValueError):
      pwlcurve.PWLCurve([(1, 1)])

  def test_duplicate_xs(self):
    with self.assertRaises(ValueError):
      pwlcurve.PWLCurve([(1, 1), (1, 1)])

  def test_unsorted_xs(self):
    with self.assertRaises(ValueError):
      pwlcurve.PWLCurve([(2, 2), (1, 1)])

  def test_eval_and_predict(self):
    curve = pwlcurve.PWLCurve([(1., 5.), (5., 13.), (10., 15.)])
    xs = [0, 1, 2, 5, 7.5, 10, 20]
    expected_ys = [5, 5, 7, 13, 14, 15, 15]
    self.assert_allclose(expected_ys, curve.eval(xs))
    self.assert_allclose(expected_ys, curve.predict(xs))

  def test_eval_with_exp_transform(self):
    orig_points = [(1., 5.), (5., 13.), (10., 15.)]
    xs = [0, 1, 2, 5, 7.5, 10, 20]

    # Shift x to exponential space.
    curve_xs, curve_ys = zip(*orig_points)
    curve_xs = np.exp(curve_xs)
    # Perform interpolation in the log-x space, counteracting the shift in x.
    curve = pwlcurve.PWLCurve(list(zip(curve_xs, curve_ys)), np.log)
    exp_x = np.exp(xs)

    expected_ys = [5, 5, 7, 13, 14, 15, 15]
    self.assert_allclose(expected_ys, curve.eval(exp_x))

  def test_eval_clamping_with_differing_float_precision(self):
    curve = pwlcurve.PWLCurve([(1.0, -0.07331), (2.0, -0.1255)], xform=np.log1p)
    xs = np.array([0., 1., 2., 3.], dtype=np.float32)

    expected_ys = [-0.07331, -0.07331, -0.1255, -0.1255]
    self.assert_allclose(expected_ys, curve.eval(xs))

  def test_round(self):
    curve = pwlcurve.PWLCurve([(1.234, 5.4321), (5.6789, 14.321)])
    rounded_curve = curve.round_to_sig_figs(2)
    self.assertEqual([(1.2, 5.4), (5.7, 14)], rounded_curve.curve_points)

  def test_round_ydigits(self):
    curve = pwlcurve.PWLCurve([(1.234, 5.4321), (5.6789, 14.321)])
    rounded_curve = curve.round_to_sig_figs(2, 4)
    self.assertEqual([(1.2, 5.432), (5.7, 14.32)], rounded_curve.curve_points)

  def test_round_large_values(self):
    curve = pwlcurve.PWLCurve([(1234, 54321), (56789, 14321)])
    rounded_curve = curve.round_to_sig_figs(2)
    self.assertEqual([(1200, 54000), (57000, 14000)],
                     rounded_curve.curve_points)

  def test_round_no_change(self):
    curve = pwlcurve.PWLCurve([(1., 5.), (5., 13.), (10., 15.)])
    rounded_curve = curve.round_to_sig_figs(2)
    self.assertEqual(curve.curve_points, rounded_curve.curve_points)

  def test_round_increases_figs_for_close_xs(self):
    curve = pwlcurve.PWLCurve([(1.23456, 5.4321), (1.23467, 6.5432),
                               (5.6789, 14.321)])
    rounded_curve = curve.round_to_sig_figs(2)
    self.assertEqual([(1.2346, 5.4), (1.2347, 6.5), (5.6789, 14)],
                     rounded_curve.curve_points)

  @parameterized.named_parameters(
      ('not too sensitive',
       pwlcurve.PWLCurve([(1.0, 1.0), (2.0, 2.0)]),
       pwlcurve.PWLCurve(([1, 1], [2, 2])),
       True),
      ('not a PWLCurve',
       pwlcurve.PWLCurve([(1.0, 1.0), (2.0, 2.0)]), 's', False),
      ('different xs',
       pwlcurve.PWLCurve([(1.0, 1.0), (2.0, 2.0)]),
       pwlcurve.PWLCurve([(10.0, 1.0), (20.0, 2.0)]),
       False),
      ('different ys',
       pwlcurve.PWLCurve([(1.0, 1.0), (2.0, 2.0)]),
       pwlcurve.PWLCurve([(1.0, 10.0), (2.0, 20.0)]),
       False),
      ('different xform',
       pwlcurve.PWLCurve([(1.0, 1.0), (2.0, 2.0)]),
       pwlcurve.PWLCurve([(1.0, 1.0), (2.0, 2.0)], np.log),
       False),
  )
  def test_eq(self, c1, c2, expected):
    self.assertEqual(c1 == c2, expected)

  @parameterized.named_parameters(
      ('simple',
       pwlcurve.PWLCurve([(1.0, 1.0), (2.0, 2.0)]),
       'PWLCurve([(1, 1), (2, 2)])'),
      ('xform',
       pwlcurve.PWLCurve([(1.0, 1.0), (2.0, 2.0)], np.log),
       'PWLCurve([(1, 1), (2, 2)], xform="log")'),
      ('rounds',
       pwlcurve.PWLCurve([(1.23456789, 1.23456789), (2.0, 2.0)]),
       'PWLCurve([(1.235, 1.235), (2, 2)])'))
  def test_str(self, curve, expected):
    self.assertEqual(str(curve), expected)

  @parameterized.named_parameters(
      ('simple', 'PWLCurve([(1.0, 1.0), (2.0, 2.0)])', transform.identity),
      ('xform', 'PWLCurve([(1.0, 1.0), (2.0, 2.0)], xform="log")', np.log),
      ('xform spaces flexible',
       'PWLCurve([(1.0, 1.0), (2.0, 2.0)]  ,  xform="log")', np.log),
      ('point syntax is flexible', 'PWLCurve(([1, 1], [2.0, 2.0]))',
       transform.identity))
  def test_from_string(self, s, xform):
    points = [(1.0, 1.0), (2.0, 2.0)]
    curve = pwlcurve.PWLCurve(points, xform)
    self.assertEqual(curve, pwlcurve.PWLCurve.from_string(s))
    self.assertEqual(curve, pwlcurve.PWLCurve.from_string(str(curve)))

  def test_from_string_bad_syntax(self):
    with self.assertRaisesRegex(ValueError, 'must begin with'):
      pwlcurve.PWLCurve.from_string('PWL Curve([(1,1),(2,2)])')

  def test_from_string_bad_xform(self):
    with self.assertRaisesRegex(ValueError, 'Invalid xform "foo"'):
      pwlcurve.PWLCurve.from_string(
          'PWLCurve([(1.0, 1.0), (2.0, 2.0)], xform="foo")')


if __name__ == '__main__':
  unittest.main()
