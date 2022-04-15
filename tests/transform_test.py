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


"""Tests for pwlfit/transform."""

import unittest


import numpy as np
from pwlfit import test_util
from pwlfit import transform
import scipy.stats


identity_transform = transform.identity
log_transform = np.log
log1p_transform = np.log1p
symlog1p_transform = transform.symlog1p


class FindBestTransformTest(test_util.PWLFitTest):

  def test_weighted_pearson_correlation_is_independent_of_weight_scale(self):
    np.random.seed(4)
    x = np.random.normal(size=100)
    y = x + np.random.normal(size=100)
    w = np.random.uniform(size=100)
    corr_w = transform.weighted_pearson_correlation(x, y, w)
    corr_5w = transform.weighted_pearson_correlation(x, y, 5 * w)
    self.assertAlmostEqual(corr_w, corr_5w)

  def test_weighted_pearson_correlation_matches_scipy_for_equal_weights(self):
    # Verify that our weighted pearson correlation matches scipy's unweighted
    # pearson correlation when the weights are equal.
    np.random.seed(4)
    x = np.random.normal(size=100)
    w = np.ones_like(x)

    # Non-correlated.
    y = np.random.normal(size=100)
    weightless_corr, _ = scipy.stats.pearsonr(x, y)
    weighted_corr = transform.weighted_pearson_correlation(x, y, w)
    self.assertAlmostEqual(weightless_corr, weighted_corr)

    # Correlated.
    y = x + np.random.normal(size=100)
    weightless_corr, _ = scipy.stats.pearsonr(x, y)
    weighted_corr = transform.weighted_pearson_correlation(x, y, w)
    self.assertAlmostEqual(weightless_corr, weighted_corr)

    # Perfect correlation.
    y = x
    weightless_corr, _ = scipy.stats.pearsonr(x, y)
    weighted_corr = transform.weighted_pearson_correlation(x, y, w)
    self.assertAlmostEqual(weightless_corr, weighted_corr)

  def test_weighted_pearson_correlation_nan_when_x_is_constant(self):
    np.random.seed(6)
    x = np.repeat(np.random.normal(), 10)
    y = np.random.normal(size=10)
    w = np.random.uniform(size=10)
    correlation = transform.weighted_pearson_correlation(x, y, w)
    self.assertTrue(np.isnan(correlation))

  def test_weighted_pearson_correlation_nan_when_y_is_constant(self):
    np.random.seed(7)
    x = np.random.normal(size=10)
    y = np.repeat(np.random.normal(), 10)
    w = np.random.uniform(size=10)
    correlation = transform.weighted_pearson_correlation(x, y, w)
    self.assertTrue(np.isnan(correlation))

  def test_weighted_pearson_correlation_raises_on_bad_input(self):
    no_pts = np.array([], dtype=float)
    two_pts = np.array([1., 2.])
    three_pts = np.array([1., 2., 3.])

    with self.assertRaises(ValueError):
      transform.weighted_pearson_correlation(no_pts, no_pts, no_pts)

    with self.assertRaises(ValueError):
      transform.weighted_pearson_correlation(three_pts, two_pts, two_pts)

    with self.assertRaises(ValueError):
      transform.weighted_pearson_correlation(three_pts, two_pts, three_pts)

    with self.assertRaises(ValueError):
      transform.weighted_pearson_correlation(three_pts, three_pts, two_pts)

    with self.assertRaises(ValueError):
      transform.weighted_pearson_correlation(no_pts, two_pts, three_pts)

    # Doesn't raise.
    transform.weighted_pearson_correlation(two_pts, two_pts, two_pts)
    transform.weighted_pearson_correlation(three_pts, three_pts, three_pts)

  def test_identity_function(self):
    np.random.seed(4)
    xs = np.random.normal(size=100)
    self.assert_allclose(xs, transform.identity(xs))

  def test_symlog1p_matches_log1p(self):
    np.random.seed(4)
    xs = np.abs(np.random.normal(size=100))
    self.assert_allclose(np.log1p(xs), transform.symlog1p(xs))
    self.assert_allclose(-np.log1p(xs), transform.symlog1p(-xs))

  def test_find_best_transform_is_identity_for_constant_xs(self):
    np.random.seed(5)
    x = np.ones(10)
    y = np.arange(10)
    w = np.random.uniform(size=len(x))
    found_transform = transform.find_best_transform(x, y, w)
    self.assertEqual(identity_transform, found_transform)

  def test_find_best_transform_is_identity_for_constant_ys(self):
    np.random.seed(5)
    x = np.arange(10)
    y = np.ones(10)
    w = np.random.uniform(size=len(x))
    found_transform = transform.find_best_transform(x, y, w, pct_to_clip=0)
    self.assertEqual(identity_transform, found_transform)

  def test_find_best_transform_is_identity_for_xs_constant_after_clipping(self):
    np.random.seed(5)
    x = np.array([1.] + [2] * 100 + [3])
    y = np.log(x)
    w = np.ones_like(x)
    found_transform = transform.find_best_transform(x, y, w, pct_to_clip=.01)
    self.assertEqual(identity_transform, found_transform)

  def test_find_best_transform_is_identity_for_ys_constant_after_clipping(self):
    np.random.seed(5)
    y = np.array([1.] + [2] * 100 + [3])
    x = np.arange(len(y))
    w = np.ones_like(x)
    found_transform = transform.find_best_transform(x, y, w, pct_to_clip=.01)
    self.assertEqual(identity_transform, found_transform)

  def test_find_best_transform_does_not_mutate_inputs(self):
    np.random.seed(5)
    x = np.sort(np.random.normal(size=123))
    y = np.random.normal(size=123)
    w = np.random.uniform(size=123)
    x_copy, y_copy, w_copy = x.copy(), y.copy(), w.copy()

    transform.find_best_transform(x, y, w)
    np.testing.assert_array_equal(x, x_copy)
    np.testing.assert_array_equal(y, y_copy)
    np.testing.assert_array_equal(w, w_copy)

  def test_find_best_transform_identity(self):
    np.random.seed(5)
    x = np.sort(np.random.uniform(high=10, size=1000))
    w = np.random.uniform(size=len(x))

    found_transform = transform.find_best_transform(x, x, w)
    self.assertEqual(identity_transform, found_transform)

    # Linear transforms are still best fit with the identity transform.
    found_transform = transform.find_best_transform(x * 97 + 5, x - 60, w)
    self.assertEqual(identity_transform, found_transform)

    found_transform = transform.find_best_transform(x / 1e5, -x / 52, w * 99)
    self.assertEqual(identity_transform, found_transform)

    # Other transforms maintain the relationship so long as they're applied to
    # both x and y.
    found_transform = transform.find_best_transform(np.exp(x), np.exp(x), w)
    self.assertEqual(identity_transform, found_transform)

    found_transform = transform.find_best_transform(np.log(x), np.log(x), w)
    self.assertEqual(identity_transform, found_transform)

    found_transform = transform.find_best_transform(x**3, x**3, w)
    self.assertEqual(identity_transform, found_transform)

  def test_find_best_transform_log_transform(self):
    np.random.seed(6)
    x = np.sort(np.random.uniform(size=1000))
    w = np.random.uniform(size=1000)

    found_transform = transform.find_best_transform(x, np.log(x), w)
    self.assertEqual(log_transform, found_transform)

    found_transform = transform.find_best_transform(np.exp(x), x, w)
    self.assertEqual(log_transform, found_transform)

  def test_find_best_transform_log1p_transform(self):
    np.random.seed(7)
    x = np.sort(np.random.uniform(-1, 1, size=1000))
    w = np.random.uniform(size=1000)

    found_transform = transform.find_best_transform(x, np.log1p(x), w)
    self.assertEqual(log1p_transform, found_transform)

    found_transform = transform.find_best_transform(np.expm1(x), x, w)
    self.assertEqual(log1p_transform, found_transform)

  def test_find_best_transform_symlog1p_transform(self):
    np.random.seed(8)
    # symlog1p extends the log distribution to allow negative inputs.
    x = np.sort(np.random.uniform(low=-4, high=5, size=1000))
    w = np.random.uniform(size=1000)
    y = transform.symlog1p(x)

    found_transform = transform.find_best_transform(x, y, w)
    self.assertEqual(symlog1p_transform, found_transform)

  def test_find_best_transform_clips_by_weight(self):
    # Generate data that's mostly linear but with a logarithmic tail.
    np.random.seed(5)
    x = np.linspace(1, 10, num=1000)
    y = np.array(x, copy=True)
    x[-5:] = 2 ** x[-5:]

    # If all weights are equal, we clip the log-tail, so the data is linear.
    w = np.ones_like(x)
    found_transform = transform.find_best_transform(x, y, w, .005)
    self.assertEqual(identity_transform, found_transform)

    # If the tail is heavy, it will dominate even after clipping.
    w = np.ones_like(x)
    w[-5:] = 10000
    found_transform = transform.find_best_transform(x, y, w, .005)
    self.assertEqual(log_transform, found_transform)


if __name__ == '__main__':
  unittest.main()
