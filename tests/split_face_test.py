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


class SplitFaceTest(FscadTestCase):
    def test_basic_split_face(self):
        box = Box(1, 1, 1)
        cylinder = Cylinder(1, .25)
        cylinder.place(~cylinder == ~box,
                       ~cylinder == ~box,
                       -cylinder == +box)

        split = SplitFace(box, cylinder)
        split.create_occurrence(True)
        self.assertEqual(len(split.split_faces), 1)
        self.assertEqual(split.split_faces[0].brep.pointOnFace.z, 1)

    def test_basic_split_face_direct(self):
        box = Box(1, 1, 1)
        cylinder = Cylinder(1, .25)
        cylinder.place(~cylinder == ~box,
                       ~cylinder == ~box,
                       -cylinder == +box)

        split = SplitFace(box, cylinder)
        split.create_occurrence(True)
        self.assertEqual(len(split.split_faces), 1)
        self.assertEqual(split.split_faces[0].brep.pointOnFace.z, 1)

    def test_multiple_lump_split_face(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        box2.place((-box2 == +box1) + 1)
        union = Union(box1, box2)

        cylinder = Cylinder(1, .25)
        cylinder.place(~cylinder == ~box1,
                       ~cylinder == ~box1,
                       -cylinder == +box1)
        split = SplitFace(union, cylinder)
        split.create_occurrence(True)
        self.assertEqual(len(split.split_faces), 1)
        self.assertEqual(split.split_faces[0].brep.pointOnFace.z, 1)
        self.assertLess(split.split_faces[0].brep.pointOnFace.x, 1)

    def test_multiple_body_split_face(self):
        rect1 = Rect(1, 1, "rect1")
        rect2 = Rect(1, 1, "rect2")
        rect2.place((-rect2 == +rect1) + 1)

        union = Union(rect1, rect2)
        extrude = Extrude(union, 1)

        cylinder = Cylinder(1, .25)
        cylinder.place(~cylinder == ~rect1,
                       ~cylinder == ~rect1,
                       (-cylinder == +rect1)+1)
        split = SplitFace(extrude, cylinder)
        split.create_occurrence(True)
        self.assertEqual(len(split.split_faces), 1)
        self.assertEqual(split.split_faces[0].brep.pointOnFace.z, 1)
        self.assertLess(split.split_faces[0].brep.pointOnFace.x, 1)

    def test_split_face_with_face(self):
        box1 = Box(1, 1, 1)
        box2 = Box(1, 1, 1)
        box2.place((-box2 == +box1) + 1)
        union = Union(box1, box2)

        cylinder = Cylinder(1, .25).ry(90)
        cylinder.place(-cylinder == +box1,
                       ~cylinder == ~box1,
                       ~cylinder == ~box1)

        split = SplitFace(union, cylinder.bottom)
        split.create_occurrence(True)
        self.assertEqual(len(split.split_faces), 1)
        self.assertEqual(split.split_faces[0].brep.pointOnFace.x, 1)
        self.assertEqual(split.split_faces[0].size().asArray(), (0, .5, .5))

    def test_split_face_with_non_coincident_body(self):
        box = Box(1, 1, 1)
        cylinder = Cylinder(1, .25)
        cylinder.place(~cylinder == ~box,
                       ~cylinder == ~box,
                       ~cylinder == +box)

        split = SplitFace(box, cylinder)
        split.create_occurrence(True)
        self.assertEqual(len(split.split_faces), 1)
        self.assertEqual(split.split_faces[0].brep.pointOnFace.z, 1)


def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__],
                                                                #pattern="multiple_body_split_face",
                                                                )
    unittest.TextTestRunner(failfast=True).run(test_suite)
