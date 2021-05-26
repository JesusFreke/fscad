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
from adsk.core import Point3D

import unittest

# note: load_tests is required for the "pattern" test filtering functionality in loadTestsFromModule in run()
from fscad.test_utils import FscadTestCase, load_tests
from fscad.fscad import *


class PlaceTest(FscadTestCase):
    def test_place(self):
        box1 = Box(1, 2, 3, "box1")
        box2 = Box(5, 6, 7, "box2")

        box2.place(-box2 == ~box1,
                   ~box2 == +box1,
                   +box2 == -box1)

        self.assertEqual(box2.size().asArray(), (5, 6, 7))
        self.assertEqual(box2.min().asArray(), (.5, 2 - 6/2, -7))
        self.assertEqual(box2.mid().asArray(), (.5 + 5/2, 2, -7/2))
        self.assertEqual(box2.max().asArray(), (.5 + 5, 2 + 6/2, 0))

        box1.create_occurrence()
        box2.create_occurrence()

    def test_place_offset(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        box2.place(
            (-box2 == +box1) + 1,
            ~box2 == ~box1,
            ~box2 == ~box1)
        box1.create_occurrence()
        box2.create_occurrence()

    def test_place_children(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        box2.place(
            (-box2 == +box1),
            ~box2 == ~box1,
            ~box2 == ~box1)
        union = Union(box1, box2)

        box3 = Box(1, 1, 1, "box3")

        union.place(-union == +box3,
                    ~union == ~box3,
                    ~union == ~box3)
        union = Union(*union.children(), box3)

        self.assertEqual(box1.size().asArray(), (1, 1, 1))
        self.assertEqual(box1.min().asArray(), (1, 0, 0))
        self.assertEqual(box1.mid().asArray(), (1.5, .5, .5))
        self.assertEqual(box1.max().asArray(), (2, 1, 1))

        union.create_occurrence(True)

    def test_place_at_value(self):
        box1 = Box(1, 1, 1, "box1")
        box1.place(
            -box1 == 1,
            ~box1 == Point3D.create(1, 2, 3),
            +box1 == 1)
        box1.create_occurrence()

        self.assertEqual(box1.size().asArray(), (1, 1, 1))
        self.assertEqual(box1.min().asArray(), (1, 1.5, 0))
        self.assertEqual(box1.mid().asArray(), (1.5, 2, .5))
        self.assertEqual(box1.max().asArray(), (2, 2.5, 1))

    def test_place_at_point(self):
        box1 = Box(1, 1, 1, "box1")
        point = box1.mid()
        point.x += .1
        point.y += .1
        point.z += .1
        box1.add_named_point("off_center", point)

        box2 = Box(1, 1, 1, "box2")
        box2.add_named_point("max_point", box2.max())
        box2.place(~box2 == ~box1.named_point("off_center"),
                   ~box2 == ~box1.named_point("off_center"),
                   ~box2 == ~box1.named_point("off_center"))
        box2.add_named_point("min_point", box2.min())

        box3 = Box(1, 1, 1, "box3")
        box3.place(-box3 == ~box2.named_point("max_point"),
                   -box3 == ~box2.named_point("max_point"),
                   -box3 == ~box2.named_point("max_point"))

        box4 = Box(1, 1, 1, "box4")
        box4.place(+box4 == ~box2.named_point("min_point"),
                   +box4 == ~box2.named_point("min_point"),
                   +box4 == ~box2.named_point("min_point"))

        box1.create_occurrence()
        box2.create_occurrence()
        box3.create_occurrence()
        box4.create_occurrence()


def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
