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


class PlaceTest(test_utils.FscadTestCase):
    def test_simple_place(self):
        first = box(1, 1, 1, name="first")
        second = place(box(1, 1, 1, name="second"),
                       midAt(atMin(first)),
                       minAt(atMax(first)),
                       maxAt(atMid(first)))

    def test_rotated_place(self):
        first = rz(box(1, 1, 1, name="first"), -45)
        second = place(rz(box(1, 1, 1, name="second"), -45),
                       minAt(atMax(first)), keep(), keep())

    def test_modified_constraint_place(self):
        first = box(1, 1, 1, name="first")
        second = place(box(1, 1, 1, name="second"),
                       midAt(atMid(first)),
                       midAt(atMid(first)),
                       minAt(lambda coord: atMax(first)(coord) + 2))

    def test_simple_place_sketch(self):
        first = rect(1, 1, name="first")
        second = place(rect(1, 1, name="second"),
                       midAt(atMin(first)),
                       minAt(atMax(first)),
                       maxAt(atMid(first)))

    def test_place_tilted_sketch(self):
        first = rotate(rect(1, 1, name="first"), x=45)
        second = place(rotate(rect(1, 1, name="second"), x=45, z=-180),
                       minAt(atMin(first)),
                       minAt(atMax(first)),
                       keep())

    def test_get_placement(self):
        first = box(1, 1, 1, name="first")
        second = place(box(.25, .25, .25, name="second"),
                       minAt(atMax(first)), midAt(atMid(first)), maxAt(atMax(first)))
        combined = union(first, second)

        ball = sphere(.25)

        get_placement(second, maxAt(atMin(ball)), midAt(atMid(ball)), midAt(atMid(ball))).apply(combined)


    def test_pointAt(self):
        first = box(1, 1, 1, name="first")
        define_point(first, midOf(first).x, minOf(first).y, midOf(first).z, name="point")
        second = box(1, 1, 1, name="second")
        place(first,
              pointAt("point", atMax(second)),
              pointAt("point", atMax(second)),
              pointAt("point", atMax(second)))

    def test_atPoint(self):
        first = box(1, 1, 1, name="first")
        second = box(1, 1, 1, name="second")
        define_point(second, midOf(second).x, minOf(second).y, midOf(second).z, name="point")

        place(first,
              maxAt(atPoint(second, "point")),
              maxAt(atPoint(second, "point")),
              maxAt(atPoint(second, "point")))

    def test_pointAtAtPoint(self):
        first = box(1, 1, 1, name="first")
        define_point(first, maxOf(first).x, midOf(first).y, maxOf(first).z, name="point")
        second = box(1, 1, 1, name="second")
        define_point(second, midOf(second).x, minOf(second).y, minOf(second).z, name="point")

        place(first,
              pointAt("point", atPoint(second, "point")),
              pointAt("point", atPoint(second, "point")),
              pointAt("point", atPoint(second, "point")))

    def test_translatedPoint(self):
        first = box(1, 1, 1, name="first")
        translate(first, 5, 5, 5)
        define_point(first, midOf(first).x, minOf(first).y, midOf(first).z, name="point")
        second = box(1, 1, 1, name="second")
        place(first,
              pointAt("point", atMax(second)),
              pointAt("point", atMax(second)),
              pointAt("point", atMax(second)))



from test_utils import load_tests
def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
