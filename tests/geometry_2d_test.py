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


class Geometry2DTest(test_utils.FscadTestCase):
    def test_rect(self):
        rect(1, 2, name="MyRectangle")

    def test_circle(self):
        circle(1, name="MyCircle")

    def test_basic_polygon(self):
        polygon(
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1))

    def test_polygon(self):
        polygon(
            (0, 0),
            (.5, 0),
            (.5, .5),
            (1, 0),
            (1, 1),
            (.25, .75),
            (0, 1),
            (.25, .5))

    def test_regular_polygons(self):
        for i in range(3, 10):
            tx(regular_polygon(i, 1), (i-3)*2)

    def test_regular_polygon_width(self):
        hexagon = regular_polygon(6, regular_polygon_radius_for_width(1, 6))
        hexagon_rect = place(rect(1, 1), midAt(atMid(hexagon)), midAt(atMid(hexagon)))

        heptagon = regular_polygon(7, regular_polygon_radius_for_width(1, 7))
        heptagon_rect = place(rect(1, 1), midAt(atMid(heptagon)), midAt(atMid(heptagon)))
        tx(heptagon, 2)
        tx(heptagon_rect, 2)


from test_utils import load_tests
def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
