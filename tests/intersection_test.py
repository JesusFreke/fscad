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


class IntersectionTest(FscadTestCase):
    def test_basic_intersection(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        box2.place(-box2 == ~box1,
                   ~box2 == ~box1,
                   ~box2 == ~box1)
        Intersection(box1, box2).create_occurrence(True)

    def test_disjoint_intersection(self):
        box1 = Box(1, 1, 1, name="box1")
        box2 = Box(1, 1, 1, name="box2")
        box2.place((-box2 == +box1) + 1,
                   ~box2 == ~box1,
                   ~box2 == ~box1)
        Intersection(box1, box2).create_occurrence(True)

    def test_adjoining_intersection(self):
        box1 = Box(1, 1, 1, name="box1")
        box2 = Box(1, 1, 1, name="box2")
        box2.place((-box2 == +box1),
                   ~box2 == ~box1,
                   ~box2 == ~box1)
        Intersection(box1, box2).create_occurrence(True)

    def test_complete_intersection(self):
        box1 = Box(1, 1, 1, name="box1")
        box2 = Box(1, 1, 1, name="box2")
        Intersection(box1, box2).create_occurrence(True)

    def test_complex_intersection(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(.5, 10, 10, "box2")
        box2.place(~box2 == ~box1,
                   ~box2 == ~box1,
                   ~box2 == ~box1)
        difference1 = Difference(box1, box2, name="difference1")

        box3 = first = Box(1, 1, 1, "box3")
        box4 = Box(10, 10, .5, "box4")
        box4.place(~box4 == ~box1,
                   ~box4 == ~box1,
                   ~box4 == ~box1)
        difference2 = Difference(box3, box4, name="difference2")
        Intersection(difference1, difference2).create_occurrence(True)

    def test_empty_intersection(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        difference = Difference(box1, box2)
        box3 = Box(1, 1, 1, "box3")
        Intersection(difference, box3).create_occurrence(True)

    def test_basic_planar_intersection(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        rect2.place(-rect2 == ~rect1, ~rect2 == ~rect1, ~rect2 == ~rect1)
        intersection = Intersection(rect1, rect2)
        intersection.create_occurrence(True)
        self.assertTrue(rect1.get_plane().isCoPlanarTo(intersection.get_plane()))

    def test_disjoint_planar_intersection(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        rect2.place((-rect2 == +rect1) + 1, ~rect2 == ~rect1, ~rect2 == ~rect1)
        intersection = Intersection(rect1, rect2)
        intersection.create_occurrence(True)
        self.assertTrue(rect1.get_plane().isCoPlanarTo(intersection.get_plane()))

    def test_adjoining_planar_intersection(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        rect2.place(-rect2 == +rect1, ~rect2 == ~rect1, ~rect2 == ~rect1)
        intersection = Intersection(rect1, rect2)
        intersection.create_occurrence(True)
        self.assertTrue(rect1.get_plane().isCoPlanarTo(intersection.get_plane()))

    def test_complete_planar_intersection(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        intersection = Intersection(rect1, rect2)
        intersection.create_occurrence(True)
        self.assertTrue(rect1.get_plane().isCoPlanarTo(intersection.get_plane()))

    def test_complex_planar_intersection(self):
        rect1 = Rect(1, 3, "rect1")
        rect2 = Rect(1, 3, "rect2")
        rect2.place((-rect2 == +rect1) + 1,
                    ~rect2 == ~rect1,
                    ~rect2 == ~rect1)
        union1 = Union(rect1, rect2, name="union1")

        rect3 = Rect(3, 1, "rect3")
        rect4 = Rect(3, 1, "rect3")
        rect4.place(~rect4 == ~rect3,
                    (-rect4 == +rect3) + 1,
                    ~rect4 == ~rect3)
        union2 = Union(rect3, rect4, name="union2")
        intersection = Intersection(union1, union2)
        intersection.create_occurrence(True)
        self.assertTrue(rect1.get_plane().isCoPlanarTo(intersection.get_plane()))

    def test_planar_3d_intersection(self):
        rect = Rect(1, 1)
        box = Box(1, 1, 1)
        box.place(-box == ~rect)
        intersection = Intersection(rect, box)
        intersection.create_occurrence(True)
        self.assertTrue(rect.get_plane().isCoPlanarTo(intersection.get_plane()))

    def test_3d_planar_intersection(self):
        rect = Rect(1, 1)
        box = Box(1, 1, 1)
        box.place(-box == ~rect)
        intersection = Intersection(box, rect)
        intersection.create_occurrence(True)
        self.assertTrue(rect.get_plane().isCoPlanarTo(intersection.get_plane()))

    def test_planar_intersection_with_empty(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        rect2.place(-rect2 == +rect1)
        empty_intersection = Intersection(rect1, rect2, name="empty_intersection")

        rect3 = Rect(1, 1, "rect3")
        intersection = Intersection(rect3, empty_intersection)
        intersection.create_occurrence(True)
        self.assertTrue(rect1.get_plane().isCoPlanarTo(intersection.get_plane()))

    def test_non_coplanar_intersection(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        rect2.place(z=(~rect2 == ~rect1) + 1)
        try:
            Intersection(rect1, rect2)
            self.fail("Expected error did not occur")
        except ValueError:
            pass

    def test_add_planar_intersection(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        rect2.place(-rect2 == ~rect1)

        intersection = Intersection(rect1, rect2)

        rect3 = Rect(1, 1, "rect3")
        rect3.place(y=-rect2 == ~rect1)
        intersection = Intersection(*intersection.children(), rect3)
        intersection.create_occurrence(True)
        self.assertTrue(rect1.get_plane().isCoPlanarTo(intersection.get_plane()))

    def test_add_planar_to_3d_intersection(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        box2.place(-box2 == ~box1)

        intersection = Intersection(box1, box2)

        self.assertIsNone(intersection.get_plane())

        rect = Rect(1, 1)
        rect.place(y=-rect == ~box1)
        intersection = Intersection(*intersection.children(), rect)
        intersection.create_occurrence(True)
        self.assertTrue(rect.get_plane().isCoPlanarTo(intersection.get_plane()))

    def test_add_3d_to_planar_intersection(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        rect2.place(-rect2 == ~rect1)

        intersection = Intersection(rect1, rect2)

        box = Box(1, 1, 1)
        box.place(y=-box == ~rect1)
        intersection = Intersection(*intersection.children(), box)
        intersection.create_occurrence(True)
        self.assertTrue(rect1.get_plane().isCoPlanarTo(intersection.get_plane()))


def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]
                                                                #, pattern="named_face_after_intersection_add"
                                                                )
    unittest.TextTestRunner(failfast=True).run(test_suite)
