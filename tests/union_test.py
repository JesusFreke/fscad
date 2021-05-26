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


class UnionTest(FscadTestCase):
    def test_basic_union(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        box2.place(-box2 == +box1,
                   ~box2 == ~box1,
                   ~box2 == ~box1)
        union = Union(box1, box2)

        self.assertEqual(union.size().asArray(), (2, 1, 1))
        self.assertEqual(union.min().asArray(), (0, 0, 0))
        self.assertEqual(union.mid().asArray(), (1, .5, .5))
        self.assertEqual(union.max().asArray(), (2, 1, 1))

        union.create_occurrence()

    def test_union_children(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        box2.place(-box2 == +box1,
                   ~box2 == ~box1,
                   ~box2 == ~box1)
        union = Union(box1, box2, name="union")

        box3 = Box(1, 1, 1, "box3")
        box3.place(-box3 == +union,
                   ~box3 == ~union,
                   ~box3 == ~union)

        union2 = Union(union, box3, name="union2")

        union2.create_occurrence(True)

    def test_union_flatten(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        box2.place(-box2 == +box1,
                   ~box2 == ~box1,
                   ~box2 == ~box1)
        union = Union(box1, box2)

        box3 = Box(1, 1, 1, "box3")
        box3.place(-box3 == +union,
                   ~box3 == ~union,
                   ~box3 == ~union)
        union = Union(*union.children(), box3)

        union.create_occurrence(True)

    def test_union_subcomponent(self):
        box1 = Box(2, 1, 1, "Box1")
        box2 = Box(1, 2, 1, "Box2")
        box2.place(~box2 == ~box1,
                   -box2 == +box1,
                   ~box2 == ~box1)
        union = Union(box1, box2)
        box3 = Box(1, 2, 1, "Box3")
        box3.place(~box3 == ~box1,
                   +box3 == -box1,
                   ~box3 == ~box1)
        union2 = Union(box1, box3)
        union2.place(z=-union2 == +box1)

        union.create_occurrence(True)
        union2.create_occurrence(True)

    def test_simple_planar_union(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        rect2.place(-rect2 == +rect1, ~rect2 == ~rect1)
        union = Union(rect1, rect2)
        union.create_occurrence(True)
        self.assertTrue(rect1.get_plane().isCoPlanarTo(union.get_plane()))

    def test_overlaping_planar_union(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        rect2.place(-rect2 == ~rect1, ~rect2 == ~rect1)
        union = Union(rect1, rect2)
        union.create_occurrence(True)
        self.assertTrue(rect1.get_plane().isCoPlanarTo(union.get_plane()))

    def test_disjoint_planar_union(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        rect2.place((-rect2 == +rect1) + 1, ~rect2 == ~rect1)
        union = Union(rect1, rect2)
        union.create_occurrence(True)
        self.assertTrue(rect1.get_plane().isCoPlanarTo(union.get_plane()))

    def test_single_planar_union(self):
        rect1 = Rect(1, 1, "rect1")
        union = Union(rect1)
        union.create_occurrence(True)
        self.assertTrue(rect1.get_plane().isCoPlanarTo(union.get_plane()))

    def test_empty_planar_union(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        empty = Difference(rect1, rect2, name="empty")

        rect3 = Rect(1, 1, "rect3")
        rect3.place((-rect3 == -rect1) + 5)

        union = Union(rect3, empty, name="union")
        union.create_occurrence(True)
        self.assertTrue(rect1.get_plane().isCoPlanarTo(union.get_plane()))

    def test_non_coplanar_union(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        rect2.place(-rect2 == +rect1,
                    ~rect2 == ~rect1,
                    (~rect2 == ~rect1) + 1)

        try:
            Union(rect1, rect2)
            self.fail("Expected error did not occur")
        except ValueError:
            pass

    def test_3D_planar_union(self):
        rect1 = Rect(1, 1, "rect1")
        box1 = Box(1, 1, 1, "box1")

        try:
            Union(rect1, box1)
            self.fail("Expected error did not occur")
        except ValueError:
            pass

        try:
            Union(box1, rect1)
            self.fail("Expected error did not occur")
        except ValueError:
            pass

    def test_planar_union_flatten(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        rect2.place(-rect2 == +rect1, ~rect2 == ~rect1)
        union = Union(rect1, rect2)

        rect3 = Rect(1, 1, "rect3")
        rect3.place(-rect3 == +rect2, ~rect3 == ~rect2)
        union = Union(*union.children(), rect3)

        union.create_occurrence(True)
        self.assertTrue(rect1.get_plane().isCoPlanarTo(union.get_plane()))

    def test_non_coplanar_union_flatten(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        rect2.place(-rect2 == +rect1, ~rect2 == ~rect1)
        union = Union(rect1, rect2)

        rect3 = Rect(1, 1, "rect3")
        rect3.place(-rect3 == +rect2, ~rect3 == ~rect2, (~rect3 == ~rect2) + 1)
        try:
            union = Union(*union.children(), rect3)
            self.fail("Expected error did not occur")
        except ValueError:
            pass

    def test_flatten_3D_to_planar_union(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        rect2.place(-rect2 == +rect1, ~rect2 == ~rect1)
        union = Union(rect1, rect2)

        box = Box(1, 1, 1)
        try:
            union = Union(*union.children(), box)
            self.fail("Expected error did not occur")
        except ValueError:
            pass

    def test_flatten_planar_to_3D_union(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        box2.place(-box2 == +box1, ~box2 == ~box1)
        union = Union(box1, box2)

        rect = Rect(1, 1)
        try:
            union = Union(*union.children(), rect)
            self.fail("Expected error did not occur")
        except ValueError:
            pass


def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
