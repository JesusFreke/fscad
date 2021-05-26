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

import math
import unittest

# note: load_tests is required for the "pattern" test filtering functionality in loadTestsFromModule in run()
from fscad.test_utils import FscadTestCase, load_tests
from fscad.fscad import *


class LoftTest(FscadTestCase):

    def test_basic_loft(self):
        rect = Rect(1, 1)
        circle = Circle(1)
        circle.place(~circle == ~rect,
                     ~circle == ~rect,
                     (~circle == ~rect) + 1)

        loft = Loft(rect, circle)
        loft.create_occurrence(True)

        self.assertEqual(loft.bottom.brep.pointOnFace.z, 0)
        self.assertTrue(math.isclose(loft.top.brep.pointOnFace.z, 1))
        self.assertEqual(len(list(loft.sides)), 4)

    def test_loft_with_hole(self):
        outer = Circle(2, "outer")
        inner = Circle(1, "inner")
        bottom = Difference(outer, inner, name="bottom")

        outer2 = Circle(1, "outer2")
        inner2 = Circle(.5, "inner2")
        top = Difference(outer2, inner2)
        top.place(~top == ~bottom,
                  ~top == ~bottom,
                  (~top == ~bottom) + 1)

        loft = Loft(bottom, top)
        loft.create_occurrence(True)

        self.assertEqual(loft.bottom.brep.pointOnFace.z, 0)
        self.assertEqual(loft.top.brep.pointOnFace.z, 1)
        self.assertEqual(len(list(loft.sides)), 1)

    def test_triple_loft(self):
        rect1 = Rect(1, 1, "rect1")
        circle = Circle(1)
        circle.place(~circle == ~rect1,
                     ~circle == ~rect1,
                     (~circle == ~rect1) + 1)
        rect2 = Rect(1, 1, "rect2")
        rect2.place(~rect2 == ~circle,
                    ~rect2 == ~circle,
                    (~rect2 == ~circle) + 1)

        loft = Loft(rect1, circle, rect2)
        loft.create_occurrence(True)

        self.assertEqual(loft.bottom.brep.pointOnFace.z, 0)
        self.assertEqual(loft.top.brep.pointOnFace.z, 2)
        self.assertEqual(len(list(loft.sides)), 4)


def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
