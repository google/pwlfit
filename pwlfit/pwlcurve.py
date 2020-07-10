# Lint as: python2, python3
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


"""Classes and helpers for representing curves."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pwlfit import transform
from pwlfit import utils


def _tosigfigs(x, num_figs):
  """Rounds x to the specified number of significant figures."""
  num_figs_str = '%d' % num_figs
  return float(('%.' + num_figs_str + 'g') % x)


# TODO(walkerravina): Refactor the rest of the library to use this class
# (e.g. have fitter.fit_pwl return instances of this class).
class PWLCurve(object):
  """An immutable class for representing a piecewise linear curve."""

  def __init__(self, curve_points, transform_fn=transform.identity):
    """Initializer.

    Args:
      curve_points: (iterable((float, float))): x,y control poins.
      transform_fn: (callable(iterable(floats)) --> iterable(floats)): Transform
        to perform before linear iterpolation.
    """
    utils.expect(
        len(curve_points) >= 2, 'A PWLCurve must have at least two knots.')
    curve_xs, curve_ys = zip(*curve_points)
    curve_xs, curve_ys = np.asarray(curve_xs), np.asarray(curve_ys)
    utils.expect(
        len(set(curve_xs)) == len(curve_xs), 'Curve knot xs must be unique.')
    utils.expect((np.sort(curve_xs) == curve_xs).all(),
                 'Curve knot xs must be ordered.')
    self._curve_xs = curve_xs
    self._curve_ys = curve_ys
    self._transform_fn = transform_fn

  @property
  def curve_points(self):
    return list(zip(self._curve_xs, self._curve_ys))

  @property
  def curve_xs(self):
    return list(self._curve_xs)

  @property
  def curve_ys(self):
    return list(self._curve_ys)

  @property
  def transform_fn(self):
    return self._transform_fn

  def eval(self, xs):
    """Returns the result of evaluating the PWLCurve on the given xs.

    Args:
      xs: (numpy array) Data to evaluate the curve on.

    Returns:
      A numpy array of the PWLCurve's value at each x in xs.
    """
    curve_xs = self._curve_xs
    curve_ys = self._curve_ys
    # Clamp the inputs to the range of the control points.
    xs = np.clip(xs, curve_xs[0], curve_xs[-1])
    if self._transform_fn is not None:
      xs = self._transform_fn(xs)
      curve_xs = self._transform_fn(curve_xs)

    indices = curve_xs.searchsorted(xs)

    # Extend curve for convenient boundary handling.
    curve_xs = np.concatenate([[curve_xs[0] - 1], curve_xs])
    curve_ys = np.concatenate([[curve_ys[0]], curve_ys])

    prev_x, prev_y = curve_xs[indices], curve_ys[indices]
    next_x, next_y = curve_xs[indices + 1], curve_ys[indices + 1]
    gap = next_x - prev_x

    return next_y * ((xs - prev_x) / gap) + prev_y * ((next_x - xs) / gap)

  def predict(self, xs):
    """Alias for eval()."""
    return self.eval(xs)

  def round_to_sig_figs(self, xfigures, yfigures=None):
    """Returns a new PWLCurve rounded to specified significant figures.

    A valid curve can't have duplicate control point xs. If the rounded curve
    has duplicate xs, we increment xfigures until the xs are no longer
    duplicates.

    Args:
      xfigures: (int) Minimum number of decimal digits to keep. For example,
        ndigits=2 rounds to 2 decimal digits (1.234 --> 1.2).
      yfigures: (int): How many decimal digits to keep for y coordinates of
        points If not set, will use xfigures.

    Returns:
      A new PWLCurve rounded to the specified number of significant figures.
    """
    if yfigures is None:
      yfigures = xfigures

    rounded_xs = [_tosigfigs(x, xfigures) for x in self._curve_xs]
    rounded_ys = [_tosigfigs(y, yfigures) for y in self._curve_ys]
    while len(rounded_xs) != len(set(rounded_xs)):
      xfigures += 1
      rounded_xs = [_tosigfigs(x, xfigures) for x in self._curve_xs]

    return PWLCurve(list(zip(rounded_xs, rounded_ys)), self._transform_fn)

  def __repr__(self):
    return 'PWLCurve(%s, %s)' % (list(zip(self._curve_xs,
                                          self._curve_ys)), self._transform_fn)

  def __str__(self):
    if self._transform_fn is transform.identity:
      return 'PWLCurve(%s)' % self.round_to_sig_figs(4).curve_points
    return 'PWLCurve(%s, %s)' % (self.round_to_sig_figs(4).curve_points,
                                 self._transform_fn.__name__)
