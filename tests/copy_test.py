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
from adsk.core import Vector3D

import unittest

# note: load_tests is required for the "pattern" test filtering functionality in loadTestsFromModule in run()
from fscad.test_utils import FscadTestCase, load_tests
from fscad.fscad import *


class CopyTest(FscadTestCase):
    def test_box_copy(self):
        box = Box(1, 1, 1)
        box_copy = box.copy()
        box_copy.place(
            -box_copy == +box,
            ~box_copy == ~box,
            ~box_copy == ~box)

        box.create_occurrence()
        box_copy.create_occurrence()

    def test_union_copy(self):
        box = Box(1, 1, 1)
        box2 = Box(1, 1, 1, "Box2")
        box2.place(-box2 == +box,
                   ~box2 == ~box,
                   ~box2 == ~box)
        union = Union(box, box2)

        box3 = Box(1, 1, 1, "Box3")
        box3.place(-box3 == +box2,
                   ~box3 == ~box2,
                   ~box3 == ~box2)

        union2 = Union(union, box3, name="Union2")

        union2_copy = union2.copy()
        union2_copy.place((-union2_copy == +union2) + 1,
                          ~union2_copy == ~union2,
                          ~union2_copy == ~union2)

        union2.create_occurrence(True)
        union2_copy.create_occurrence(True)

    def test_shallow_copy(self):
        box = Box(1, 1, 1)
        box2 = Box(1, 1, 1, "Box2")
        box2.place(-box2 == +box,
                   ~box2 == ~box,
                   ~box2 == ~box)
        union = Union(box, box2)

        box3 = Box(1, 1, 1, "Box3")
        box3.place(-box3 == +box2,
                   ~box3 == ~box2,
                   ~box3 == ~box2)

        union2 = Union(union, box3, name="Union2")

        union2_copy = union2.copy(False)
        union2_copy.place((-union2_copy == +union2) + 1,
                          ~union2_copy == ~union2,
                          ~union2_copy == ~union2)

        union2.create_occurrence(True)
        union2_copy.create_occurrence(True)

    def test_rotated_planar_union_shallow_copy(self):
        rect = Rect(1, 1, name="rect")
        rect2 = Rect(1, 1, name="rect2")

        rect2.place(-rect2 == +rect,
                    ~rect2 == ~rect,
                    ~rect2 == ~rect)
        union = Union(rect, rect2)
        union_copy = union.copy(False)
        union_rotated_copy = union.copy(False)
        union_rotated_copy.ry(45)

        union.create_occurrence(True)
        union_copy.create_occurrence(True)
        union_rotated_copy.create_occurrence(True)

        self.assertIsNotNone(union.get_plane())
        self.assertIsNotNone(union_copy.get_plane())
        self.assertIsNotNone(union_rotated_copy.get_plane())
        self.assertEqual(union.get_plane().uDirection.asArray(), union_copy.get_plane().uDirection.asArray())
        self.assertEqual(union.get_plane().vDirection.asArray(), union_copy.get_plane().vDirection.asArray())

        self.assertTrue(union_rotated_copy.get_plane().normal.isParallelTo(Vector3D.create(1, 0, 1)))

    def test_rotated_planar_difference_shallow_copy(self):
        rect = Rect(1, 1, name="rect")
        box = Box(.5, .5, .5, name="box")

        box.place(~box == ~rect,
                  ~box == ~rect,
                  ~box == ~rect)

        diff = Difference(rect, box)

        diff_copy = diff.copy(False)
        diff_rotated_copy = diff.copy(False)
        diff_rotated_copy.ry(45)

        diff.create_occurrence(True)
        diff_copy.create_occurrence(True)
        diff_rotated_copy.create_occurrence(True)

        self.assertIsNotNone(diff.get_plane())
        self.assertIsNotNone(diff_copy.get_plane())
        self.assertIsNotNone(diff_rotated_copy.get_plane())
        self.assertEqual(diff.get_plane().uDirection.asArray(), diff_copy.get_plane().uDirection.asArray())
        self.assertEqual(diff.get_plane().vDirection.asArray(), diff_copy.get_plane().vDirection.asArray())

        self.assertTrue(diff_rotated_copy.get_plane().normal.isParallelTo(Vector3D.create(1, 0, 1)))

    def test_rotated_planar_intersection_shallow_copy(self):
        box = Box(.5, .5, .5, name="box")
        rect = Rect(1, 1, name="rect")

        rect.place(~rect == ~box,
                   ~rect == ~box,
                   ~rect == ~box)

        intersection = Intersection(box, rect)

        intersection_copy = intersection.copy(False)
        intersection_rotated_copy = intersection.copy(False)
        intersection_rotated_copy.ry(45)

        intersection.create_occurrence(True)
        intersection_copy.create_occurrence(True)
        intersection_rotated_copy.create_occurrence(True)

        self.assertIsNotNone(intersection.get_plane())
        self.assertIsNotNone(intersection_copy.get_plane())
        self.assertIsNotNone(intersection_rotated_copy.get_plane())
        self.assertEqual(intersection.get_plane().uDirection.asArray(), intersection_copy.get_plane().uDirection.asArray())
        self.assertEqual(intersection.get_plane().vDirection.asArray(), intersection_copy.get_plane().vDirection.asArray())

        self.assertTrue(intersection_rotated_copy.get_plane().normal.isParallelTo(Vector3D.create(1, 0, 1)))

    def test_split_copy(self):
        box = Box(1, 1, 1)
        cyl = Cylinder(1, .5)
        cyl.place(~cyl == ~box,
                  ~cyl == ~box,
                  -cyl == +box)
        split = SplitFace(box, cyl)

        split2 = split.copy()
        split2.place((-split2 == +split) + 1)

        split3 = split2.copy()
        split3.place((-split3 == +split2) + 1)

        split.create_occurrence()
        split2.create_occurrence()
        split3.create_occurrence()


def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
