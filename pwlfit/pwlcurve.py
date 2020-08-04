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
import ast
from typing import Callable, List, Sequence, Tuple
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

  STR_TO_XFORM = {
      fn.__name__: fn for fn in
      [transform.identity, transform.symmetriclog1p, np.log, np.log1p]
  }

  def __init__(self,
               curve_points: Sequence[Tuple[float, float]],
               xform: Callable[[np.ndarray], np.ndarray] = transform.identity):
    """Initializer.

    Args:
      curve_points: x,y control points.
      xform: Transform to apply to x values before linear interpolation.
    """
    utils.expect(
        len(curve_points) >= 2, 'A PWLCurve must have at least two knots.')
    curve_xs, curve_ys = zip(*curve_points)
    curve_xs, curve_ys = (np.asarray(curve_xs, dtype=float),
                          np.asarray(curve_ys, dtype=float))
    utils.expect(
        len(set(curve_xs)) == len(curve_xs), 'Curve knot xs must be unique.')
    utils.expect((np.sort(curve_xs) == curve_xs).all(),
                 'Curve knot xs must be ordered.')
    self._curve_xs = curve_xs
    self._curve_ys = curve_ys
    self._xform = xform

  @classmethod
  def from_string(cls, s: str) -> 'PWLCurve':
    """Parses a PWLCurve from the given string.

    Syntax is that emitted by __str__: PWLCurve({points}, xform="{xform_name}")

    Only xforms present in STR_TO_XFORM will be parsed.

    Args:
      s: The string to parse.

    Returns:
      The parsed PWLCurve.
    """
    prefix = 'PWLCurve('
    utils.expect(
        s.startswith(prefix) and s.endswith(')'),
        'String must begin with "%s" and end with ")"' % prefix)
    s = s[len(prefix):-1]
    idx = s.find('xform=')
    if idx < 0:
      return cls(ast.literal_eval(s))
    xform_str = ast.literal_eval(s[idx + len('xform='):])
    xform = cls.STR_TO_XFORM.get(xform_str)
    utils.expect(xform is not None, 'Invalid xform "%s" specified' % xform_str)
    control_points = ast.literal_eval(s[:s.rfind(',')].rstrip())
    return cls(control_points, xform)

  @property
  def curve_points(self) -> List[Tuple[float, float]]:
    return list(zip(self._curve_xs, self._curve_ys))

  @property
  def curve_xs(self) -> List[float]:
    return list(self._curve_xs)

  @property
  def curve_ys(self) -> List[float]:
    return list(self._curve_ys)

  @property
  def xform(self) -> Callable[[np.ndarray], np.ndarray]:
    return self._xform

  def eval(self, xs) -> np.ndarray:
    """Returns the result of evaluating the PWLCurve on the given xs.

    Args:
      xs: (numpy array) Data to evaluate the curve on.

    Returns:
      A numpy array of the PWLCurve's value at each x in xs.
    """
    curve_xs = self._curve_xs
    curve_ys = self._curve_ys
    # Clamp the inputs to the range of the control points.
    xs = np.array(xs, dtype=curve_xs.dtype, copy=False)
    xs = np.clip(xs, curve_xs[0], curve_xs[-1])
    if self._xform is not None:
      xs = self._xform(xs)
      curve_xs = self._xform(curve_xs)

    indices = curve_xs.searchsorted(xs)

    # Extend curve for convenient boundary handling.
    curve_xs = np.concatenate([[curve_xs[0] - 1], curve_xs])
    curve_ys = np.concatenate([[curve_ys[0]], curve_ys])

    prev_x, prev_y = curve_xs[indices], curve_ys[indices]
    next_x, next_y = curve_xs[indices + 1], curve_ys[indices + 1]
    gap = next_x - prev_x

    return next_y * ((xs - prev_x) / gap) + prev_y * ((next_x - xs) / gap)

  def predict(self, xs) -> np.ndarray:
    """Alias for eval()."""
    return self.eval(xs)

  def round_to_sig_figs(self,
                        xfigures: int,
                        yfigures: int = None) -> 'PWLCurve':
    """Returns a new PWLCurve rounded to specified significant figures.

    A valid curve can't have duplicate control point xs. If the rounded curve
    has duplicate xs, we increment xfigures until the xs are no longer
    duplicates.

    Args:
      xfigures: Minimum number of decimal digits to keep. For example, ndigits=2
        rounds to 2 decimal digits (1.234 --> 1.2).
      yfigures: How many decimal digits to keep for y coordinates of points If
        not set, will use xfigures.

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

    return PWLCurve(list(zip(rounded_xs, rounded_ys)), self._xform)

  def __eq__(self, o) -> bool:
    return (isinstance(o, PWLCurve) and self.curve_xs == o.curve_xs and
            self.curve_ys == o.curve_ys and self.xform == o.xform)

  def __repr__(self):
    return 'PWLCurve(%s, %s)' % (list(zip(self._curve_xs,
                                          self._curve_ys)), self._xform)

  def __str__(self):
    points_str = '[%s]' % ', '.join(
        '(%g, %g)' % (x, y)
        for (x, y) in self.round_to_sig_figs(4).curve_points)
    if self._xform is transform.identity:
      return 'PWLCurve(%s)' % points_str
    return 'PWLCurve(%s, xform="%s")' % (points_str, self._xform.__name__)
