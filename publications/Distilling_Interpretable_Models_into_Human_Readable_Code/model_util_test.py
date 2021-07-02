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
"""Tests for model_util."""
import math
import unittest
from absl.testing import parameterized

import numpy as np
import pandas as pd
from pwlfit import fitter
from pwlfit import test_util
from pwlfit import transform
import model_util


class EnumCurveModelTest(test_util.PWLFitTest, parameterized.TestCase):

  def test_simple_properties(self):
    model = model_util.EnumCurveModel('x', {1: 1.5, 2: 2.5})
    self.assertEqual('x', model.feature_name)
    self.assertEqual('f_x', model.name)
    self.assertEqual('f_x', model.clean_name)

  @parameterized.named_parameters((
      'not too sensitive',
      model_util.EnumCurveModel('x', {
          1: 1,
          2: 2
      }),
      model_util.EnumCurveModel('x', {
          1: 1.0,
          2: 2.0
      }),
      True,
  ), (
      'not a EnumCurveModel',
      model_util.EnumCurveModel('x', {
          1: 1.0,
          2: 2.0
      }),
      's',
      False,
  ), (
      'different mappings',
      model_util.EnumCurveModel('x', {
          1: 1.0,
          2: 2.0
      }),
      model_util.EnumCurveModel('x', {
          1: 1.5,
          2: 2.5
      }),
      False,
  ))
  def test_eq(self, m1, m2, expected):
    self.assertEqual(expected, m1 == m2)

  def test_eval(self):
    model = model_util.EnumCurveModel('x', {1: 1.5, 2: 2.5})
    self.assert_allclose(
        np.array([1.5, 1.5, 2.5, 2.5]), model.eval([1, 1, 2, 2]))
    self.assert_allclose(
        np.array([1.5, 1.5, 2.5, 2.5]),
        model.eval(pd.DataFrame({'x': [1, 1, 2, 2]})))

  def test_expr(self):
    model = model_util.EnumCurveModel('x', {1: 1.5, 2: 2.5})
    self.assertEqual('EnumCurve("x", {1: 1.5, 2: 2.5})', model.expr())

  @parameterized.named_parameters(
      ('simple', 'EnumCurve("x", {1: 1.5, 2: 2.5})'),
      ('extra spaces', 'EnumCurve( "x"   ,   {1:  1.5, 2:  2.5})'))
  def test_from_expr(self, s):
    model = model_util.EnumCurveModel('x', {1: 1.5, 2: 2.5})
    self.assertEqual(model, model_util.EnumCurveModel.from_expr(s))


class PWLCurveModelTest(test_util.PWLFitTest, parameterized.TestCase):

  def test_simple_properties(self):
    model = model_util.PWLCurveModel('x', [(1, 10.0), (2, 20.0)])
    self.assertEqual('x', model.feature_name)
    self.assertEqual('f_x', model.name)
    self.assertEqual('f_x', model.clean_name)

  @parameterized.named_parameters((
      'not too sensitive',
      model_util.PWLCurveModel('x', [(1, 10.0), (2, 20.0)]),
      model_util.PWLCurveModel('x', ((1, 10), (2, 20))),
      True,
  ), (
      'not a PWLCurveModel',
      model_util.PWLCurveModel('x', [(1, 10.0), (2, 20.0)]),
      's',
      False,
  ), (
      'different points',
      model_util.PWLCurveModel('x', [(1, 10.0), (2, 20.0)]),
      model_util.PWLCurveModel('x', [(1, 11.0), (2, 21.0)]),
      False,
  ))
  def test_eq(self, m1, m2, expected):
    self.assertEqual(expected, m1 == m2)

  def test_str_for_fx(self):
    self.assertEqual(
        model_util.PWLCurveModel('x', [(1, 10.0), (2, 20.0)], np.log),
        model_util.PWLCurveModel('x', [(1, 10.0), (2, 20.0)], 'log'))

  def test_eval(self):
    model = model_util.PWLCurveModel('x', [(1, 10.0), (2, 20.0)])
    self.assert_allclose(np.array([10.0, 15.0, 20.0]), model.eval([1, 1.5, 2]))
    self.assert_allclose(
        np.array([10.0, 15.0, 20.0]),
        model.eval(pd.DataFrame({'x': [1, 1.5, 2]})))

  def test_expr(self):
    model = model_util.PWLCurveModel('x', [(1, 10.0), (2, 20.0)])
    self.assertEqual('PWLCurve("x", [(1, 10), (2, 20)])', model.expr())

  @parameterized.named_parameters((
      'simple',
      'PWLCurve("x", [(1, 10.0), (2, 20.0)])',
      transform.identity,
  ), (
      'fx',
      'PWLCurve("x", [(1, 10.0), (2, 20.0)], fx="log")',
      np.log,
  ), (
      'extra spaces',
      'PWLCurve(  "x",   [(1,  10.0), (2, 20.0)])',
      transform.identity,
  ))
  def test_from_expr(self, s, fx):
    model = model_util.PWLCurveModel('x', [(1, 10.0), (2, 20.0)], fx)
    self.assertEqual(model, model_util.PWLCurveModel.from_expr(s))


class ColumnModelTest(test_util.PWLFitTest, parameterized.TestCase):

  def test_simple_properties(self):
    model = model_util.ColumnModel('x1')
    self.assertEqual('x1', model.name)
    self.assertEqual('x1', model.feature_name)

  @parameterized.named_parameters((
      'simple ',
      model_util.ColumnModel('x1'),
      model_util.ColumnModel('x1'),
      True,
  ), (
      'not a ColumnModel',
      model_util.ColumnModel('x1'),
      's',
      False,
  ), (
      'different columns',
      model_util.ColumnModel('x1'),
      model_util.ColumnModel('x2'),
      False,
  ))
  def test_eq(self, m1, m2, expected):
    self.assertEqual(expected, m1 == m2)

  def test_eval(self):
    model = model_util.ColumnModel('x1')
    self.assert_allclose(
        np.array([1.0, 2.0, 3.0]),
        model.eval(pd.DataFrame({'x1': [1.0, 2.0, 3.0]})))

  def test_expr(self):
    model = model_util.ColumnModel('x1')
    self.assertEqual('Column("x1")', model.expr())

  @parameterized.named_parameters((
      'simple',
      'Column("x1")',
  ), (
      'extra spaces',
      'Column( "x1"   )',
  ))
  def test_from_expr(self, s):
    model = model_util.ColumnModel('x1')
    self.assertEqual(model, model_util.ColumnModel.from_expr(s))


class AdditiveModelTest(test_util.PWLFitTest, parameterized.TestCase):

  def test_simple_properties(self):
    model = model_util.AdditiveModel([
        model_util.PWLCurveModel('x1', [(1, 1), (2, 2)]),
        model_util.EnumCurveModel('x2', {
            1: 1,
            2: 2
        }),
        model_util.ColumnModel('x3')
    ], 'my_model')
    self.assertEqual('my_model', model.name)
    self.assertEqual(['x1', 'x2', 'x3'], model.feature_names)

  @parameterized.named_parameters((
      'not too sensitive ',
      model_util.AdditiveModel([
          model_util.ColumnModel('x1'),
          model_util.ColumnModel('x2'),
      ], 'my_model'),
      model_util.AdditiveModel([
          model_util.ColumnModel('x2'),
          model_util.ColumnModel('x1'),
      ], 'my_model'),
      True,
  ), (
      'not an AdditiveModel',
      model_util.AdditiveModel([
          model_util.ColumnModel('x1'),
          model_util.ColumnModel('x2'),
      ], 'my_model'),
      's',
      False,
  ), (
      'different submodels',
      model_util.AdditiveModel([
          model_util.ColumnModel('x1'),
      ], 'my_model'),
      model_util.AdditiveModel([
          model_util.ColumnModel('x2'),
      ], 'my_model'),
      False,
  ), (
      'different names',
      model_util.AdditiveModel([
          model_util.ColumnModel('x1'),
      ], 'my_model1'),
      model_util.AdditiveModel([
          model_util.ColumnModel('x1'),
      ], 'my_model2'),
      False,
  ))
  def test_eq(self, m1, m2, expected):
    self.assertEqual(expected, m1 == m2)

  def test_eval(self):
    model = model_util.AdditiveModel([
        model_util.PWLCurveModel('x1', [(1, 1), (2, 2)]),
        model_util.EnumCurveModel('x2', {
            1: 10,
            2: 20
        }),
        model_util.ColumnModel('x3')
    ], 'my_model')
    df = pd.DataFrame({'x1': [1, 1, 2], 'x2': [1, 1, 2], 'x3': [3, 3, 3]})
    self.assert_allclose(np.array([14, 14, 25]), model.eval(df))

  def test_expr(self):
    model = model_util.AdditiveModel([
        model_util.PWLCurveModel('x1', [(1, 1), (2, 2)]),
        model_util.EnumCurveModel('x2', {
            1: 1,
            2: 2
        }),
        model_util.ColumnModel('x3')
    ], 'my_model')
    expr = """score = sum([
  PWLCurve("x1", [(1, 1), (2, 2)]),
  EnumCurve("x2", {1: 1, 2: 2}),
  Column("x3"),
])"""
    self.assertEqual(expr, model.expr())

  @parameterized.named_parameters((
      'simple',
      """score = sum([
        PWLCurve("x1", [(1, 1), (2, 2)]),
        EnumCurve("x2", {1: 1, 2: 2}),
        Column("x3"),
      ])
     """,
  ), (
      'flexible',
      """score = sum([PWLCurve("x1", [(1, 1), (2, 2)]),

        EnumCurve("x2", {1: 1, 2: 2}),

        Column(  "x3"  ),
      ])
     """,
  ))
  def test_from_expr(self, expr):
    model = model_util.AdditiveModel([
        model_util.PWLCurveModel('x1', [(1, 1), (2, 2)]),
        model_util.EnumCurveModel('x2', {
            1: 1,
            2: 2
        }),
        model_util.ColumnModel('x3')
    ], 'my_model')
    self.assertEqual(model,
                     model_util.AdditiveModel.from_expr(expr, 'my_model'))

  def test_no_duplicate_features_allowed(self):
    with self.assertRaisesRegex(ValueError, 'Duplicate submodels'):
      model_util.AdditiveModel([
          model_util.ColumnModel('x1'),
          model_util.PWLCurveModel('x1', [(1, 1), (2, 2)])
      ], 'my_model')


class FitAdditiveModelTest(test_util.PWLFitTest, parameterized.TestCase):

  def test_simple(self):
    df = pd.DataFrame({
        'x1': np.linspace(0, 1, 100),
        'y1': np.linspace(0, 1, 100) * 2,
        'x2': [1, 2, 3, 4] * 25,
        'y2': [2, 4, 6, 8] * 25
    })
    model, times = model_util.fit_additive_model(df, {'x1': 'y1'}, {'x2': 'y2'},
                                                 fitter.fit_pwl, 'my_model')
    self.assertEqual(['x1', 'x2'], model.feature_names)
    self.assertEqual('my_model', model.name)
    self.assert_allclose(df.y1 + df.y2, model.eval(df))
    self.assertIsInstance(model.get_submodel('x1'), model_util.PWLCurveModel)
    self.assertIsInstance(model.get_submodel('x2'), model_util.EnumCurveModel)
    self.assertCountEqual(['f_x1', 'f_x2'], times.keys())


class ApproxSampleTest(test_util.PWLFitTest):

  def test_simple(self):
    self.assertEqual(
        100, len(model_util.approx_sample(num_items=100, num_samples=10)))
    self.assertTrue(
        all(
            np.ones(100) == model_util.approx_sample(
                num_items=100, num_samples=1000)))


class NDCGMetricTest(test_util.PWLFitTest):

  def test_simple(self):

    def log2(v):
      return math.log(v, 2)
    _ = -float('inf')

    y_true = np.array([[0, 1, 3, 0, 3],
                       [3, 4, _, _, _],
                       [1, _, _, _, _]])
    y_score = np.array([[20, 30, 10, 0, -10],
                        [50, 40, _, _, _],
                        [10, _, _, _, _]])
    y_size = np.array([5, 2, 1])
    expected = [((2**1 - 1) / log2(1 + 1) + (2**0 - 1) / log2(2 + 1) +
                 (2**3 - 1) / log2(3 + 1)) /
                ((2**3 - 1) / log2(1 + 1) + (2**3 - 1) / log2(2 + 1) +
                 (2**1 - 1) / log2(3 + 1)),
                ((2**3 - 1) / log2(1 + 1) + (2**4 - 1) / log2(2 + 1)) /
                ((2**4 - 1) / log2(1 + 1) + (2**3 - 1) / log2(2 + 1)),
                1.0]
    self.assert_allclose(expected,
                         model_util.ndcg_metric(y_true, y_score, 3, y_size))


if __name__ == '__main__':
  unittest.main()
