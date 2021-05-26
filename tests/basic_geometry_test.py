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


class BasicGeometryTest(FscadTestCase):
    def test_box(self):
        box1 = Box(1, 2, 3, "box1")

        self.assertEqual(box1.size().asArray(), (1, 2, 3))
        self.assertEqual(box1.min().asArray(), (0, 0, 0))
        self.assertEqual(box1.mid().asArray(), (.5, 1, 1.5))
        self.assertEqual(box1.max().asArray(), (1, 2, 3))

        bottom = box1.bottom
        self.assertTrue(bottom.brep.geometry.normal.isParallelTo(Vector3D.create(0, 0, 1)))
        self.assertEqual(bottom.brep.pointOnFace.z, 0)

        top = box1.top
        self.assertTrue(top.brep.geometry.normal.isParallelTo(Vector3D.create(0, 0, 1)))
        self.assertEqual(top.brep.pointOnFace.z, 3)

        right = box1.right
        self.assertTrue(right.brep.geometry.normal.isParallelTo(Vector3D.create(1, 0, 0)))
        self.assertEqual(right.brep.pointOnFace.x, 1)

        left = box1.left
        self.assertTrue(left.brep.geometry.normal.isParallelTo(Vector3D.create(1, 0, 0)))
        self.assertEqual(left.brep.pointOnFace.x, 0)

        front = box1.front
        self.assertTrue(front.brep.geometry.normal.isParallelTo(Vector3D.create(0, 1, 0)))
        self.assertEqual(front.brep.pointOnFace.y, 0)

        back = box1.back
        self.assertTrue(back.brep.geometry.normal.isParallelTo(Vector3D.create(0, 1, 0)))
        self.assertEqual(back.brep.pointOnFace.y, 2)

        box1.create_occurrence()

    def test_placed_box_faces(self):
        box1 = Box(1, 2, 3, "box1")
        box2 = Box(1, 1, 1, "box2")
        box1.place(-box1 == +box2,
                   -box1 == +box2,
                   -box1 == +box2)

        bottom = box1.bottom
        self.assertTrue(bottom.brep.geometry.normal.isParallelTo(Vector3D.create(0, 0, 1)))
        self.assertEqual(bottom.brep.pointOnFace.z, 1)

        top = box1.top
        self.assertTrue(top.brep.geometry.normal.isParallelTo(Vector3D.create(0, 0, 1)))
        self.assertEqual(top.brep.pointOnFace.z, 4)

        right = box1.right
        self.assertTrue(right.brep.geometry.normal.isParallelTo(Vector3D.create(1, 0, 0)))
        self.assertEqual(right.brep.pointOnFace.x, 2)

        left = box1.left
        self.assertTrue(left.brep.geometry.normal.isParallelTo(Vector3D.create(1, 0, 0)))
        self.assertEqual(left.brep.pointOnFace.x, 1)

        front = box1.front
        self.assertTrue(front.brep.geometry.normal.isParallelTo(Vector3D.create(0, 1, 0)))
        self.assertEqual(front.brep.pointOnFace.y, 1)

        back = box1.back
        self.assertTrue(back.brep.geometry.normal.isParallelTo(Vector3D.create(0, 1, 0)))
        self.assertEqual(back.brep.pointOnFace.y, 3)

        box1.create_occurrence()

    def test_basic_cylinder(self):
        cylinder = Cylinder(1, 1)

        self.assertEqual(cylinder.size().asArray(), (2, 2, 1))
        self.assertEqual(cylinder.min().asArray(), (-1, -1, 0))
        self.assertEqual(cylinder.mid().asArray(), (0, 0, .5))
        self.assertEqual(cylinder.max().asArray(), (1, 1, 1))
        self.assertEqual(cylinder.radius, 1)
        self.assertEqual(cylinder.top_radius, 1)
        self.assertEqual(cylinder.bottom_radius, 1)
        self.assertEqual(cylinder.height, 1)
        self.assertTrue(cylinder.axis.isEqualTo(Vector3D.create(0, 0, 1)))

        bottom = cylinder.bottom
        self.assertTrue(bottom.brep.geometry.normal.isParallelTo(Vector3D.create(0, 0, 1)))
        self.assertEqual(bottom.brep.pointOnFace.z, 0)

        top = cylinder.top
        self.assertTrue(top.brep.geometry.normal.isParallelTo(Vector3D.create(0, 0, 1)))
        self.assertEqual(top.brep.pointOnFace.z, 1)

        side = cylinder.side
        self.assertTrue(isinstance(side.brep.geometry, adsk.core.Cylinder))

        self.assertEqual(cylinder.angle, 0)

        cylinder.create_occurrence()

    def test_partial_cone(self):
        cylinder = Cylinder(1, 1, .5)

        self.assertEqual(cylinder.size().asArray(), (2, 2, 1))
        self.assertEqual(cylinder.min().asArray(), (-1, -1, 0))
        self.assertEqual(cylinder.mid().asArray(), (0, 0, .5))
        self.assertEqual(cylinder.max().asArray(), (1, 1, 1))
        self.assertEqual(cylinder.radius, 1)
        self.assertEqual(cylinder.top_radius, .5)
        self.assertEqual(cylinder.bottom_radius, 1)
        self.assertEqual(cylinder.height, 1)
        self.assertTrue(cylinder.axis.isEqualTo(Vector3D.create(0, 0, 1)))

        bottom = cylinder.bottom
        self.assertTrue(bottom.brep.geometry.normal.isParallelTo(Vector3D.create(0, 0, 1)))
        self.assertEqual(bottom.brep.pointOnFace.z, 0)

        top = cylinder.top
        self.assertTrue(top.brep.geometry.normal.isParallelTo(Vector3D.create(0, 0, 1)))
        self.assertEqual(top.brep.pointOnFace.z, 1)

        side = cylinder.side
        self.assertTrue(isinstance(side.brep.geometry, adsk.core.Cone))

        cylinder.create_occurrence()

    def test_cone(self):
        cylinder = Cylinder(1, 1, 0)

        self.assertEqual(cylinder.size().asArray(), (2, 2, 1))
        self.assertEqual(cylinder.min().asArray(), (-1, -1, 0))
        self.assertEqual(cylinder.mid().asArray(), (0, 0, .5))
        self.assertEqual(cylinder.max().asArray(), (1, 1, 1))
        self.assertEqual(cylinder.radius, 1)
        self.assertEqual(cylinder.top_radius, 0)
        self.assertEqual(cylinder.bottom_radius, 1)
        self.assertEqual(cylinder.height, 1)
        self.assertTrue(cylinder.axis.isEqualTo(Vector3D.create(0, 0, 1)))

        bottom = cylinder.bottom
        self.assertTrue(bottom.brep.geometry.normal.isParallelTo(Vector3D.create(0, 0, 1)))
        self.assertEqual(bottom.brep.pointOnFace.z, 0)

        self.assertIsNone(cylinder.top)

        side = cylinder.side
        self.assertTrue(isinstance(side.brep.geometry, adsk.core.Cone))

        self.assertEqual(cylinder.angle, 45)

        cylinder.create_occurrence()

    def test_inverse_cone(self):
        cylinder = Cylinder(1, 0, 1)

        self.assertEqual(cylinder.size().asArray(), (2, 2, 1))
        self.assertEqual(cylinder.min().asArray(), (-1, -1, 0))
        self.assertEqual(cylinder.mid().asArray(), (0, 0, .5))
        self.assertEqual(cylinder.max().asArray(), (1, 1, 1))
        self.assertEqual(cylinder.radius, 0)
        self.assertEqual(cylinder.top_radius, 1)
        self.assertEqual(cylinder.bottom_radius, 0)
        self.assertEqual(cylinder.height, 1)
        self.assertTrue(cylinder.axis.isEqualTo(Vector3D.create(0, 0, 1)))

        self.assertIsNone(cylinder.bottom)

        top = cylinder.top
        self.assertTrue(top.brep.geometry.normal.isParallelTo(Vector3D.create(0, 0, 1)))
        self.assertEqual(top.brep.pointOnFace.z, 1)

        side = cylinder.side
        self.assertTrue(isinstance(side.brep.geometry, adsk.core.Cone))

        self.assertEqual(cylinder.angle, -45)

        cylinder.create_occurrence()

    def test_sphere(self):
        sphere = Sphere(1)

        self.assertEqual(sphere.size().asArray(), (2, 2, 2))
        self.assertEqual(sphere.min().asArray(), (-1, -1, -1))
        self.assertEqual(sphere.mid().asArray(), (0, 0, 0))
        self.assertEqual(sphere.max().asArray(), (1, 1, 1))

        self.assertTrue(isinstance(sphere.surface.brep.geometry, adsk.core.Sphere))

        sphere.create_occurrence()

    def test_rect(self):
        rect = Rect(2, 3)

        self.assertEqual(rect.size().asArray(), (2, 3, 0))
        self.assertEqual(rect.min().asArray(), (0, 0, 0))
        self.assertEqual(rect.mid().asArray(), (1, 1.5, 0))
        self.assertEqual(rect.max().asArray(), (2, 3, 0))

        rect.create_occurrence()

    def test_rotated_rect(self):
        rect = Rect(2, 3)
        self.assertTrue(rect.get_plane().normal.isParallelTo(Vector3D.create(0, 0, 1)))
        rect.ry(45)
        self.assertTrue(rect.get_plane().normal.isParallelTo(Vector3D.create(1, 0, 1)))

        rect.create_occurrence()

    def test_circle(self):
        circle = Circle(2)

        self.assertEqual(circle.size().asArray(), (4, 4, 0))
        self.assertEqual(circle.min().asArray(), (-2, -2, 0))
        self.assertEqual(circle.mid().asArray(), (0, 0, 0))
        self.assertEqual(circle.max().asArray(), (2, 2, 0))

        circle.create_occurrence()

    def test_circle_copy(self):
        circle = Circle(2)
        new_circle = circle.copy().tx(1)
        new_circle.create_occurrence()

    def test_polygon(self):
        Polygon((0, 0),
                (1, 0),
                (1, 1),
                (0, 1)).create_occurrence(True)

    def test_regular_polygon_outer(self):
        RegularPolygon(6, 1, True).create_occurrence(True)

    def test_regular_polygon_inner_even(self):
        RegularPolygon(6, 1, False).create_occurrence(True)

    def test_regular_polygon_inner_odd(self):
        RegularPolygon(7, 1, False).create_occurrence(True)

    def test_torus(self):
        torus = Torus(10, 1)

        self.assertEqual(torus.size().asArray(), (22, 22, 2))

        torus.create_occurrence(True)

    def test_torus_no_center(self):
        torus = Torus(2, 2)

        self.assertEqual(torus.size().asArray(), (8, 8, 4))

        torus.create_occurrence(True)

    def test_torus_self_intersecting(self):
        torus = Torus(2, 3)

        self.assertEqual(torus.size().asArray(), (10, 10, 6))

        torus.create_occurrence(True)


def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(
        sys.modules[__name__],
        # pattern="torus_self_intersecting"
    )
    unittest.TextTestRunner(failfast=True).run(test_suite)
