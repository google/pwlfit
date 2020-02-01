PWLFit is a small piecewise-linear curve fitting library.
This is not an officially supported Google product.

Example Usage

```python
import numpy as np
from pwlfit import fitter
from pwlfit import utils

# Fit a simple 1 segment line
xs = np.arange(100)
ys = xs * 2.0
points, transform = fitter.fit_pwl(xs, ys, num_segments=1,)
print('control points: ', points)
predicted_ys = utils.eval_pwl_curve(xs, points, transform)
print('MSE: ', np.sum((ys - predicted_ys) **2.0) / len(ys))


# Fit a non monotonic 2 segment line
xs = np.arange(100)
ys = np.concatenate((np.arange(50), np.arange(50, 0, -1)))
points, transform = fitter.fit_pwl(xs, ys, num_segments=2, mono=False)
print('control points: ', points)
predicted_ys = utils.eval_pwl_curve(xs, points, transform)
print('MSE: ', np.sum((ys - predicted_ys) **2.0) / len(ys))
```
