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


class RevolveTest(FscadTestCase):
    def test_full_revolve(self):
        rect = Rect(1, 1)
        rect.tx(1)

        revolve = Revolve(rect, adsk.core.Line3D.create(Point3D.create(0, 0, 0), Point3D.create(0, 1, 0)))
        revolve.create_occurrence(True)

        self.assertEquals(len(revolve.faces), 4)

    def test_partial_revolve(self):
        rect = Rect(1, 1)
        rect.tx(1)

        revolve = Revolve(rect, adsk.core.Line3D.create(Point3D.create(0, 0, 0), Point3D.create(0, 1, 0)), 180)
        revolve.create_occurrence(True)

        self.assertEquals(len(revolve.faces), 6)

    def test_partial_revolve_negative(self):
        rect = Rect(1, 1)
        rect.tx(1)

        revolve = Revolve(rect, adsk.core.Line3D.create(Point3D.create(0, 0, 0), Point3D.create(0, 1, 0)), -180)
        revolve.create_occurrence(True)

        self.assertEquals(len(revolve.faces), 6)

    def test_revolve_off_axis(self):
        rect = Rect(1, 1)
        rect.tx(1)

        revolve = Revolve(rect, adsk.core.Line3D.create(Point3D.create(0, 0, 0), Point3D.create(0, 1, 1)))
        revolve.create_occurrence(True)

        self.assertEquals(len(revolve.faces), 4)

    def test_revolve_around_edge(self):
        rect = Rect(1, 1)

        revolve = Revolve(rect, adsk.core.Line3D.create(Point3D.create(0, 0, 0), Point3D.create(0, 1, 0)))

        revolve.create_occurrence(True)

        self.assertEquals(len(revolve.faces), 3)

    def test_revolve_face(self):
        box = Box(1, 1, 1)
        box.tx(1)

        revolve = Revolve(box.top, adsk.core.Line3D.create(Point3D.create(0, 0, 1), Point3D.create(0, 1, 1)), -180)
        revolve.create_occurrence(True)

    def test_revolve_multiple_face(self):
        box1 = Box(1, 1, 1)
        box1.tx(1)

        box2 = Box(1, 1, 1)
        box2.place(
            (-box2 == +box1) + 1,
            ~box2 == ~box1,
            ~box2 == ~box1)

        assembly = Union(box1, box2)

        revolve = Revolve(assembly.find_faces((box1.top, box2.top)),
                          adsk.core.Line3D.create(Point3D.create(0, 0, 1), Point3D.create(0, 1, 1)), -180)
        revolve.create_occurrence(True)

    def test_revolve_component_with_multiple_faces(self):
        rect1 = Rect(1, 1)
        rect1.tx(1)

        rect2 = Rect(1, 1)
        rect2.place(
            (-rect2 == +rect1) + 1,
            ~rect2 == ~rect1,
            ~rect2 == ~rect1)

        revolve = Revolve(Union(rect1, rect2),
                          adsk.core.Line3D.create(Point3D.create(0, 0, 0), Point3D.create(0, 1, 0)), -180)
        revolve.create_occurrence(True)

    def test_revolve_with_edge_axis(self):
        box = Box(1, 1, 1)
        upper_right_edge = box.shared_edges(box.top, box.right)[0]

        revolve = Revolve(box.top, upper_right_edge, 180)
        revolve.create_occurrence(True)

        self.assertEquals(revolve.size().asArray(), (box.size().x * 2, box.size().y, box.size().z*2))


def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__],
                                                                #pattern="revolve_component_with_multiple_faces"
                                                                )
    unittest.TextTestRunner(failfast=True).run(test_suite)
