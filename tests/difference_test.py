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


class DifferenceTest(FscadTestCase):
    def test_basic_difference(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(.5, .5, .5, "box2")
        box2.place(+box2 == +box1,
                   ~box2 == ~box1,
                   ~box2 == ~box1)
        Difference(box1, box2).create_occurrence(True)

    def test_disjoint_difference(self):
        box1 = Box(1, 1, 1, name="box1")
        box2 = Box(1, 1, 1, name="box2")
        box2.place((-box2 == +box1) + 1,
                   ~box2 == ~box1,
                   ~box2 == ~box1)
        Difference(box1, box2).create_occurrence(True)

    def test_adjoining_difference(self):
        box1 = Box(1, 1, 1, name="box1")
        box2 = Box(1, 1, 1, name="box2")
        box2.place((-box2 == +box1),
                   ~box2 == ~box1,
                   ~box2 == ~box1)
        Difference(box1, box2).create_occurrence(True)

    def test_complete_difference(self):
        box1 = Box(1, 1, 1, name="box1")
        box2 = Box(1, 1, 1, name="box2")
        Difference(box1, box2).create_occurrence(True)

    def test_complex_difference(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(.5, 10, 10, "box2")
        box2.place(~box2 == ~box1,
                   ~box2 == ~box1,
                   ~box2 == ~box1)
        difference1 = Difference(box1, box2, name="difference1")

        box3 = Box(1, 1, 1, "box3")
        box4 = Box(10, 10, .5, "box4")
        box4.place(~box4 == ~box1,
                   ~box4 == ~box1,
                   ~box4 == ~box1)

        difference2 = Difference(box3, box4, name="difference2")
        Difference(difference1, difference2, name="difference3").create_occurrence(True)

    def test_empty_difference(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        difference1 = Difference(box1, box2, name="difference1")
        box3 = Box(1, 1, 1, "box3")
        Difference(difference1, box3, name="difference2").create_occurrence(True)

    def test_simple_planar_difference(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        rect2.place(-rect2 == ~rect1)
        diff = Difference(rect1, rect2)
        diff.create_occurrence(True)
        self.assertTrue(rect1.get_plane().isCoPlanarTo(diff.get_plane()))

    def test_disjoint_planar_difference(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        rect2.place((-rect2 == +rect1) + 1)
        diff = Difference(rect1, rect2)
        diff.create_occurrence(True)
        self.assertTrue(rect1.get_plane().isCoPlanarTo(diff.get_plane()))

    def test_adjoining_planar_difference(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        rect2.place(-rect2 == +rect1)
        diff = Difference(rect1, rect2)
        diff.create_occurrence(True)
        self.assertTrue(rect1.get_plane().isCoPlanarTo(diff.get_plane()))

    def test_complete_planar_difference(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        diff = Difference(rect1, rect2)
        diff.create_occurrence(True)
        self.assertTrue(rect1.get_plane().isCoPlanarTo(diff.get_plane()))

    def test_empty_planar_difference(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        diff = Difference(rect1, rect2)
        rect3 = Rect(1, 1, "rect3")
        diff2 = Difference(diff, rect3, name="diff2")
        diff2.create_occurrence(True)
        self.assertTrue(rect1.get_plane().isCoPlanarTo(diff2.get_plane()))

    def test_complex_sketch_difference(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(.5, 1, "rect2")
        rect2.place((-rect2 == -rect1) + .25)
        diff1 = Difference(rect1, rect2, name="diff1")

        rect3 = Rect(1, 1, "rect3")
        rect4 = Rect(1, .5, "rect4")
        rect4.place(y=(-rect4 == -rect1) + .25)
        diff2 = Difference(rect3, rect4, name="diff2")

        diff3 = Difference(diff1, diff2, name="diff3")
        diff3.create_occurrence(True)
        self.assertTrue(rect1.get_plane().isCoPlanarTo(diff3.get_plane()))

    def test_diff_3D_from_planar(self):
        rect = Rect(1, 1)
        box = Box(1, 1, 1)
        box.place(-box == ~rect)
        diff = Difference(rect, box)
        diff.create_occurrence(True)
        self.assertTrue(rect.get_plane().isCoPlanarTo(diff.get_plane()))

    def test_planar_from_3D(self):
        rect = Rect(1, 1)
        box = Box(1, 1, 1)
        box.place(-box == ~rect)
        try:
            diff = Difference(box, rect)
            self.fail("Expected error did not occur")
        except ValueError:
            pass

    def test_inside_hole_planar_difference(self):
        outer = Rect(1, 1, "outer")
        inner = Rect(.5, .5, "inner")
        inner.place(~inner == ~outer, ~inner == ~outer, ~inner == ~outer)
        diff = Difference(outer, inner)
        diff.create_occurrence(True)
        self.assertTrue(outer.get_plane().isCoPlanarTo(diff.get_plane()))

    def test_non_coplanar_difference(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        rect2.place(z=(~rect2 == ~rect1) + 1)
        try:
            diff1 = Difference(rect1, rect2, name="diff1")
            self.fail("Expected error did not occur")
        except ValueError:
            pass

    def test_difference_with_inside_hole(self):
        inner = Rect(1, 1, "inner")
        inner_hole = Rect(.5, .5, "inner_hole")
        inner_hole.place(~inner_hole == ~inner, ~inner_hole == ~inner, ~inner_hole == ~inner)
        diff1 = Difference(inner, inner_hole, name="diff1")

        outer = Rect(2, 2, "outer")
        outer.place(~outer == ~inner, ~outer == ~inner, ~outer == ~inner)
        diff2 = Difference(outer, diff1, name="diff2")
        diff2.create_occurrence(True)
        self.assertTrue(inner.get_plane().isCoPlanarTo(diff2.get_plane()))

    def test_difference_add_planar(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        rect2.place(-rect2 == ~rect1)
        diff = Difference(rect1, rect2)

        rect3 = Rect(1, 1, "rect3")
        rect3.place(y=+rect3 == ~rect1)
        diff = Difference(*diff.children(), rect3)
        diff.create_occurrence(True)
        self.assertTrue(rect1.get_plane().isCoPlanarTo(diff.get_plane()))

    def test_add_3D_to_planar_difference(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        rect2.place(-rect2 == ~rect1)
        diff = Difference(rect1, rect2)

        box = Box(1, 1, 1)
        box.place(y=+box == ~rect1)
        diff = Difference(*diff.children(), box)
        diff.create_occurrence(True)
        self.assertTrue(rect1.get_plane().isCoPlanarTo(diff.get_plane()))

    def test_add_planar_to_3D_difference(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        box2.place(-box2 == ~box1)
        diff = Difference(box1, box2)

        rect = Rect(1, 1)
        rect.place(y=+rect == ~box1)
        try:
            diff = Difference(*diff.children(), rect)
            self.fail("Expected error did not occur")
        except ValueError:
            pass

    def test_add_non_coplanar(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        rect2.place(-rect2 == ~rect1)
        diff = Difference(rect1, rect2)

        rect3 = Rect(1, 1, "rect3")
        rect3.place(y=+rect3 == ~rect1, z=(~rect3 == ~rect1) + 1)
        try:
            diff = Difference(*diff.children(), rect3)
            self.fail("Expected error did not occur")
        except ValueError:
            pass


def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]
                                                                #, pattern="named_face_after_difference_add"
                                                                )
    unittest.TextTestRunner(failfast=True).run(test_suite)
