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
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from pwlfit import transform
from pwlfit import utils


def _tosigfigs(x, num_figs):
  """Rounds x to the specified number of significant figures."""
  num_figs_str = '%d' % num_figs
  return float(('%.' + num_figs_str + 'g') % x)


class PWLCurve(object):
  """An immutable class for representing a piecewise linear curve."""

  STR_TO_FX = {
      fn.__name__: fn for fn in
      [transform.identity, transform.symlog1p, np.log, np.log1p]
  }

  def __init__(self,
               points: Sequence[Tuple[float, float]],
               fx: Callable[[np.ndarray], np.ndarray] = transform.identity):
    """Initializer.

    Args:
      points: x,y control points.
      fx: Transform to apply to x values before linear interpolation.
    """
    utils.expect(len(points) >= 2, 'A PWLCurve must have at least two knots.')
    curve_xs, curve_ys = zip(*points)
    curve_xs, curve_ys = (np.asarray(curve_xs, dtype=float),
                          np.asarray(curve_ys, dtype=float))
    utils.expect(
        len(set(curve_xs)) == len(curve_xs), 'Curve knot xs must be unique.')
    utils.expect((np.sort(curve_xs) == curve_xs).all(),
                 'Curve knot xs must be ordered.')
    self._curve_xs = curve_xs
    self._curve_ys = curve_ys
    self._fx = fx

  @classmethod
  def from_string(cls, s: str) -> 'PWLCurve':
    """Parses a PWLCurve from the given string.

    Syntax is that emitted by __str__: PWLCurve({points}, fx="{fx_name}")

    Only fxs present in STR_TO_FX will be parsed.

    Args:
      s: The string to parse.

    Returns:
      The parsed PWLCurve.
    """
    prefix = 'PWLCurve('
    utils.expect(
        s.startswith(prefix) and s.endswith(')'),
        'String must begin with "%s" and end with ")"' % prefix)
    s = s[len(prefix) - 1:]
    idx = s.find('fx=')
    if idx < 0:
      return cls(ast.literal_eval(s))
    fx_str = ast.literal_eval(s[idx + len('fx='):-1])
    fx = cls.STR_TO_FX.get(fx_str)
    utils.expect(fx is not None, 'Invalid fx "%s" specified' % fx_str)
    control_points = ast.literal_eval(s[:s.rfind(',')] + ')')
    return cls(control_points, fx)

  @property
  def points(self) -> List[Tuple[float, float]]:
    return list(zip(self._curve_xs, self._curve_ys))

  @property
  def xs(self) -> List[float]:
    return list(self._curve_xs)

  @property
  def ys(self) -> List[float]:
    return list(self._curve_ys)

  @property
  def fx(self) -> Callable[[np.ndarray], np.ndarray]:
    return self._fx

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
    if self._fx is not None:
      xs = self._fx(xs)
      curve_xs = self._fx(curve_xs)

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
                        yfigures: Optional[int] = None) -> 'PWLCurve':
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

    return PWLCurve(list(zip(rounded_xs, rounded_ys)), self._fx)

  def __eq__(self, o) -> bool:
    return (isinstance(o, PWLCurve) and self.points == o.points and
            self.fx == o.fx)

  def __repr__(self):
    return 'PWLCurve(%s, %s)' % (list(zip(self._curve_xs,
                                          self._curve_ys)), self._fx)

  def __str__(self):
    points_str = '[%s]' % ', '.join(
        '(%g, %g)' % (x, y) for (x, y) in self.points)
    if self._fx is transform.identity:
      return 'PWLCurve(%s)' % points_str
    return 'PWLCurve(%s, fx="%s")' % (points_str, self._fx.__name__)
