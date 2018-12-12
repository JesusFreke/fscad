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
import test_utils
import importlib
importlib.reload(test_utils)
import test_utils

from fscad import *


class ThreadTest(test_utils.FscadTestCase):
    def test_square_external_thread(self):
        cyl = cylinder(20, 5)

        points = [
            (0, 0),
            (.5, 0),
            (.5, .5),
            (0, .5),
        ]

        threaded_cyl = union(cyl, threads(cyl, points, 1), name="threaded_cylinder")

    def test_square_internal_thread(self):
        outer_cyl = cylinder(20, 10, name="outer")
        inner_cyl = cylinder(20, 5, name="inner")
        outer = difference(outer_cyl, inner_cyl)

        points = [
            (0, 0),
            (.5, 0),
            (.5, .5),
            (0, .5),
        ]

        threaded_hole = difference(outer, threads(faces(outer, faces(inner_cyl, "side"))[0], points, 1),
                                   name="threaded_hole")

    def test_trapezoidal_external_thread(self):
        cyl = cylinder(20, 5)

        points = [
            (0, 0),
            (.25, .25),
            (.25, .5),
            (0, .75),
        ]

        threaded_cyl = union(cyl, threads(cyl, points, 1), name="threaded_cylinder")

    def test_trapezoidal_internal_thread(self):
        outer_cyl = cylinder(20, 10, name="outer")
        inner_cyl = cylinder(20, 5, name="inner")
        outer = difference(outer_cyl, inner_cyl)

        points = [
            (0, 0),
            (.25, .25),
            (.25, .5),
            (0, .75),
        ]

        threaded_hole = difference(outer, threads(faces(outer, faces(inner_cyl, "side"))[0], points, 1),
                                   name="threaded_hole")


from test_utils import load_tests
def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
