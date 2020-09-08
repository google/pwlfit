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


"""Setup script for pwlfit."""

import setuptools

setuptools.setup(
    name="pwlfit",
    version="0.1.0",
    author="Ranklab Team",
    author_email="noreply@google.com",
    packages=["pwlfit"],
    url="https://github.com/google/pwlfit",
    license="Apache-2.0",
    description="Library for fitting piecewise linear curves.",
    install_requires=[
        "numpy",
        "scipy",
    ],
    tests_require=[
        "nose",
    ],
)
