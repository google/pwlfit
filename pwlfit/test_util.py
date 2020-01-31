# Lint as: python2, python3
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


"""Testing helpers for pwlfit."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np


class PWLFitTest(unittest.TestCase):

  def assert_allclose(self, first, second):
    # np.testing.assert_allclose has no absolute tolerance by default.
    # Consequently, it returns false negatives when the correct value is 0.
    np.testing.assert_allclose(first, second, atol=1e-10)

  def assert_notallclose(self, first, second):
    np.testing.assert_raises(
        AssertionError, np.testing.assert_allclose, first, second, atol=1e-10)

  def assert_increasing(self, seq):
    for a, b in zip(seq[:-1], seq[1:]):
      self.assertLessEqual(a, b)

  def assert_decreasing(self, seq):
    for a, b in zip(seq[:-1], seq[1:]):
      self.assertGreaterEqual(a, b)



