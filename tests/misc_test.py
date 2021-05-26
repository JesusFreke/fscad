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

import math
import random
import unittest

# note: load_tests is required for the "pattern" test filtering functionality in loadTestsFromModule in run()
from fscad.test_utils import FscadTestCase, load_tests
from fscad.fscad import *
import fscad.fscad


class MiscTest(FscadTestCase):
    def validate_test(self):
        pass

    def _do_project_point_to_line_test(self, point: adsk.core.Point3D, line: adsk.core.InfiniteLine3D):
        projection = fscad.fscad._project_point_to_line(point, line)
        self.assertTrue(projection.isEqualTo(point) or line.direction.isPerpendicularTo(projection.vectorTo(point)))
        self.assertTrue(projection.isEqualTo(line.origin) or
                        projection.vectorTo(line.origin).isParallelTo(line.direction))

    def test_project_point_to_line_vertical(self):
        self._do_project_point_to_line_test(
            adsk.core.Point3D.create(1, 0, 0),
            adsk.core.InfiniteLine3D.create(
                adsk.core.Point3D.create(0, 0, 0),
                adsk.core.Vector3D.create(0, 0, 1)))

        self._do_project_point_to_line_test(
            adsk.core.Point3D.create(1, 0, 0),
            adsk.core.InfiniteLine3D.create(
                adsk.core.Point3D.create(0, 0, 100),
                adsk.core.Vector3D.create(0, 0, 1)))

        self._do_project_point_to_line_test(
            adsk.core.Point3D.create(1, 0, 0),
            adsk.core.InfiniteLine3D.create(
                adsk.core.Point3D.create(0, 0, 100),
                adsk.core.Vector3D.create(0, 0, -1)))

        self._do_project_point_to_line_test(
            adsk.core.Point3D.create(1, 0, 44732),
            adsk.core.InfiniteLine3D.create(
                adsk.core.Point3D.create(0, 0, 100),
                adsk.core.Vector3D.create(0, 0, 1)))

        self._do_project_point_to_line_test(
            adsk.core.Point3D.create(0, 0, 0),
            adsk.core.InfiniteLine3D.create(
                adsk.core.Point3D.create(0, 0, 100),
                adsk.core.Vector3D.create(0, 0, 1)))

    def test_project_point_to_line_horizontal(self):
        self._do_project_point_to_line_test(
            adsk.core.Point3D.create(0, 0, 1),
            adsk.core.InfiniteLine3D.create(
                adsk.core.Point3D.create(0, 0, 0),
                adsk.core.Vector3D.create(1, 0, 0)))

        self._do_project_point_to_line_test(
            adsk.core.Point3D.create(0, 0, 1),
            adsk.core.InfiniteLine3D.create(
                adsk.core.Point3D.create(100, 0, 0),
                adsk.core.Vector3D.create(1, 0, 0)))

        self._do_project_point_to_line_test(
            adsk.core.Point3D.create(0, 0, 1),
            adsk.core.InfiniteLine3D.create(
                adsk.core.Point3D.create(100, 0, 0),
                adsk.core.Vector3D.create(-1, 0, 0)))

        self._do_project_point_to_line_test(
            adsk.core.Point3D.create(44732, 0, 1),
            adsk.core.InfiniteLine3D.create(
                adsk.core.Point3D.create(100, 0, 0),
                adsk.core.Vector3D.create(-1, 0, 0)))

    def test_project_point_to_line_angle(self):
        self._do_project_point_to_line_test(
            adsk.core.Point3D.create(1, 1, 0),
            adsk.core.InfiniteLine3D.create(
                adsk.core.Point3D.create(0, 0, 0),
                adsk.core.Vector3D.create(1, 1, 1)))

        self._do_project_point_to_line_test(
            adsk.core.Point3D.create(1, 1, 0),
            adsk.core.InfiniteLine3D.create(
                adsk.core.Point3D.create(64, 2, 337),
                adsk.core.Vector3D.create(1, 1, 1)))

        self._do_project_point_to_line_test(
            adsk.core.Point3D.create(44732, 0, 1),
            adsk.core.InfiniteLine3D.create(
                adsk.core.Point3D.create(100, 100, 100),
                adsk.core.Vector3D.create(1, 1, 1)))

    def test_get_arbitrary_perpendicular_unit_vector(self):
        vector = adsk.core.Vector3D.create(1, 2, 3)
        perpendicular = fscad.fscad._get_arbitrary_perpendicular_unit_vector(vector)
        self.assertTrue(perpendicular.isPerpendicularTo(vector))

    def test_closest_points(self):
        box1 = Box(1, 1, 1, name="box1")
        box2 = Box(1, 1, 1, name="box2")
        box2.place(~box2 == ~box1,
                   (-box2 == +box1) + 2,
                   -box2 == -box1)

        (point1, point2) = box1.closest_points(box2)
        self.assertEqual(point1.distanceTo(point2), 2)
        self.assertEqual(box1.bodies[0].brep.pointContainment(point1),
                         adsk.fusion.PointContainment.PointOnPointContainment)
        self.assertEqual(box2.bodies[0].brep.pointContainment(point2),
                         adsk.fusion.PointContainment.PointOnPointContainment)

    def test_thickness_box(self):
        box = Box(1, 1, 1)

        self.assertEqual(box.thickness(Vector3D.create(1, 0, 0)), 1)
        self.assertEqual(box.thickness(Vector3D.create(0, 1, 0)), 1)
        self.assertEqual(box.thickness(Vector3D.create(0, 0, 1)), 1)
        self.assertAlmostEqual(box.thickness(Vector3D.create(1, 1, 0)), math.sqrt(2))
        self.assertAlmostEqual(box.thickness(Vector3D.create(1, 1, 1)), math.sqrt(3))

    def test_thickness_sphere(self):
        sphere = Sphere(1)
        for i in range(0, 10):
            self.assertAlmostEqual(
                sphere.thickness(Vector3D.create(random.random(), random.random(), random.random())),
                2)

    def test_xy_bounding_box(self):
        circle1 = Circle(1)
        circle2 = Circle(1)
        circle2.place(
            -circle2 == +circle1)
        combined = Union(circle1, circle2)
        bounding_box = combined.bounding_box.make_box()

        self.assertTrue(bounding_box.get_plane().isCoPlanarTo(circle1.get_plane()))
        self.assertEqual(bounding_box.size().asArray(), (4, 2, 0))

    def test_yz_bounding_box(self):
        circle1 = Circle(1)
        circle2 = Circle(1)
        circle2.place(
            -circle2 == +circle1)
        combined = Union(circle1, circle2)
        combined.ry(90)
        bounding_box = combined.bounding_box.make_box()

        bounding_box.create_occurrence(scale=.1)
        combined.create_occurrence(scale=.1)

        self.assertTrue(bounding_box.get_plane().isCoPlanarTo(circle1.get_plane()))
        self.assertEqual(bounding_box.size().asArray(), (0, 2, 4))

    def test_xz_bounding_box(self):
        circle1 = Circle(1)
        circle2 = Circle(1)
        circle2.place(
            -circle2 == +circle1)
        combined = Union(circle1, circle2)
        combined.rx(90)
        bounding_box = combined.bounding_box.make_box()

        self.assertTrue(bounding_box.get_plane().isCoPlanarTo(circle1.get_plane()))
        self.assertEqual(bounding_box.size().asArray(), (4, 0, 2))


def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
