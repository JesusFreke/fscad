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
from adsk.core import Point3D, Vector3D

import unittest

# note: load_tests is required for the "pattern" test filtering functionality in loadTestsFromModule in run()
from fscad.test_utils import FscadTestCase, load_tests
from fscad.fscad import *


class AlignTest(FscadTestCase):
    def test_box_align(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        box2.place((-box2 == +box1) + 1)

        box2.align_to(box1, Vector3D.create(-1, 0, 0))

        box1.create_occurrence(True)
        box2.create_occurrence(True)

    def test_align_to_multiple_bodies(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        box3 = Box(1, 1, 1, "box3")

        box2.place(
            (+box2 == -box1) - 1,
            ~box2 == ~box1,
            ~box2 == ~box1)

        box3.place((-box3 == +box1) + 1)

        assembly = Group([box1, box2])

        box3.align_to(assembly, Vector3D.create(-1, 0, 0))

        assembly.create_occurrence(True)
        box3.create_occurrence(True)

    def test_rotated_box_align(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        box1.rz(15)
        box2.rz(-15)
        box2.place((-box2 == +box1) + 1)

        box2.align_to(box1, Vector3D.create(-1, 0, 0))

        box1.create_occurrence(True)
        box2.create_occurrence(True)

    def test_sphere_align(self):
        sphere1 = Sphere(1, "sphere1")
        sphere2 = Sphere(1, "sphere2")
        sphere2.ty(5).tx(.5)

        sphere1.align_to(sphere2, Vector3D.create(0, 1, 0))
        sphere1.create_occurrence(True)
        sphere2.create_occurrence(True)

    def test_bounding_boxes_intersect_but_geometry_doesnt(self):
        box1 = Box(2, .5, .5, "box1")
        box1_addition = Box(.5, .5, 2)
        box1_addition.place(+box1_addition == +box1,
                            +box1_addition == +box1,
                            -box1_addition == -box1)
        box1 = Union(box1, box1_addition)

        box2 = box1.copy().ry(180, center=box1.mid())
        box2.tx(-1).ty(10)

        try:
            box1.align_to(box2, Vector3D.create(0, 1, 0))
            self.fail("Expected error did not occur")
        except ValueError:
            pass

    def test_align_to_point(self):
        sphere = Sphere(1)
        point = Point3D.create(5, .5, .5)
        sphere.align_to(Point(point), Vector3D.create(1, 0, 0))
        sphere.add_named_point("point", point)
        sphere.create_occurrence(True)
        self.assertEqual(sphere.bodies[0].brep.pointContainment(point),
                         adsk.fusion.PointContainment.PointOnPointContainment)

    def test_align_to_body(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        box2.place((-box2 == +box1) + 1)

        box2.align_to(box1.bodies[0], Vector3D.create(-1, 0, 0))

        box1.create_occurrence(True)
        box2.create_occurrence(True)

    def test_align_to_face(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        box1.rz(25)
        box2.rz(-15)
        box2.place((-box2 == +box1) + 1)

        box2.align_to(box1.back, Vector3D.create(-1, 0, 0))

        box1.create_occurrence(True)
        box2.create_occurrence(True)


def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(
        sys.modules[__name__]
        #, pattern="align_to_multiple_bodies"
        )
    unittest.TextTestRunner(failfast=True).run(test_suite)
