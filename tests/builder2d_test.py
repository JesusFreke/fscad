# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import adsk.fusion

import unittest

# note: load_tests is required for the "pattern" test filtering functionality in loadTestsFromModule in run()
from fscad.test_utils import FscadTestCase, load_tests
from fscad.fscad import *


class Builder2DTest(FscadTestCase):
    def test_line_to(self):
        builder = Builder2D((0, 0))
        builder.line_to((0, 1))
        builder.line_to((1, 1))
        builder.line_to((1, 0))
        builder.line_to((0, 0))
        builder.build().create_occurrence()

    def test_spline_line(self):
        builder = Builder2D((0, 0))
        builder.fit_spline_through((0, 1))
        builder.fit_spline_through((1, 1))
        builder.fit_spline_through((1, 0))
        builder.fit_spline_through((0, 0))
        builder.build().create_occurrence()

    def test_spline(self):
        builder = Builder2D((0, 0))
        builder.fit_spline_through((.75, 1.25), (2, 2))
        builder.line_to((builder.last_point.x, 0))
        builder.line_to((0, 0))
        builder.build().create_occurrence()

    def test_spline_loop(self):
        builder = Builder2D((0, 0))
        builder.fit_spline_through((0, 1), (1, 1), (1, 0), (0, 0))
        builder.build().create_occurrence()


def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__],
                                                                # pattern="spline_loop",
                                                                )
    unittest.TextTestRunner(failfast=True).run(test_suite)
