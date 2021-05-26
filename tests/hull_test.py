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


class HullTest(FscadTestCase):
    def test_concave_quadrilateral(self):
        points = [
            Point3D.create(0, 0, 0),
            Point3D.create(1, 1, 0),
            Point3D.create(-1, 0, 0),
            Point3D.create(1, -1, 0)]
        hull = Hull(Polygon(*points))
        hull.create_occurrence(True)

        self.assertEqual(len(hull.edges), 3)

    def test_separate_equal_circles(self):
        circle1 = Circle(1)
        circle2 = Circle(1)
        circle2.tx(3)

        hull = Hull(Union(circle1, circle2))
        hull.create_occurrence(True)

        self.assertEqual(len(hull.edges), 4)

    def test_single_circle(self):
        hull = Hull(Circle(1))
        hull.create_occurrence(True)

        self.assertEqual(len(hull.edges), 1)

    def test_separate_dissimilar_circles(self):
        circle1 = Circle(10)
        circle2 = Circle(1)
        circle2.place(
            (-circle2 == +circle1) + 1,
            ~circle2 == ~circle1,
            ~circle2 == ~circle1)

        hull = Hull(Union(circle1, circle2))
        hull.create_occurrence(True)

        self.assertEqual(len(hull.edges), 4)

    def test_circles_intersection_vertical(self):
        circle1 = Circle(1)
        circle2 = Circle(1)
        circle2.tx(.9)

        hull = Hull(Intersection(circle1, circle2))
        hull.create_occurrence(True)

        self.assertEqual(len(hull.edges), 2)

    def test_circles_intersection_horizontal(self):
        circle1 = Circle(1)
        circle2 = Circle(1)
        circle2.ty(.9)

        hull = Hull(Intersection(circle1, circle2))
        hull.create_occurrence(True)

        self.assertEqual(len(hull.edges), 2)

    def test_three_circle_intersection(self):
        circle1 = Circle(1)
        circle2 = Circle(1)
        circle3 = Circle(.1)
        circle2.tx(.9)

        temp = Intersection(circle1, circle2)
        circle3.place(
            ~circle3 == ~temp,
            (-circle3 == +temp) - .025,
            ~circle3 == ~temp)

        hull = Hull(Intersection(circle1, circle2, circle3), .001)
        hull.create_occurrence(True)

        self.assertEqual(len(hull.edges), 3)
        self.assertTrue(isinstance(hull.edges[0].brep.geometry, adsk.core.Arc3D))
        self.assertTrue(isinstance(hull.edges[1].brep.geometry, adsk.core.Arc3D))
        self.assertTrue(isinstance(hull.edges[1].brep.geometry, adsk.core.Arc3D))

    def test_circle_intersection_with_concave_third_circle_union(self):
        circle1 = Circle(1)
        circle2 = Circle(1)
        circle3 = Circle(.1)
        circle2.tx(.9)

        intersection = Intersection(circle1, circle2)
        circle3.place(
            ~circle3 == ~intersection,
            (-circle3 == +intersection),
            ~circle3 == ~intersection)

        hull = Hull(Union(intersection, circle3))
        hull.create_occurrence(True)

        self.assertEqual(len(hull.edges), 5)

    def test_separate_ellipses(self):
        circle1 = Circle(1)
        circle2 = Circle(1)
        circle2.tx(3)
        xy_plane = circle1.get_plane()

        union = Union(circle1, circle2)
        union.ry(45)

        silhouette = Silhouette(union, xy_plane)

        for edge in silhouette.edges:
            self.assertTrue(isinstance(edge.brep.geometry, adsk.core.Ellipse3D) or
                            isinstance(edge.brep.geometry, adsk.core.EllipticalArc3D))

        hull = Hull(silhouette, .001)
        hull.create_occurrence(True)

    def test_separate_nurbs(self):
        box = Box(5, 5, 1)
        fillet = Fillet(box.shared_edges(box.front, [box.left, box.right]), 1)
        xy_plane = box.bottom.get_plane()

        fillet.tz(10)
        fillet.ry(45)
        fillet.rx(30)

        fillet_bottom = fillet.find_faces(box.bottom)[0]
        upper_silhouette = Silhouette(fillet_bottom, box.bottom.get_plane())
        lower_silhouette = Silhouette(fillet_bottom, xy_plane)
        loft = Loft(lower_silhouette, upper_silhouette)

        loft2 = loft.copy()
        loft2.place(
            (-loft2 == +loft) + 5,
            (~loft2 == ~loft) + 5,
            -loft2 == -loft)

        union = Union(loft, loft2)
        bottom_finder = Box(*union.size().asArray())
        bottom_finder.place(
            ~bottom_finder == ~union,
            ~bottom_finder == ~union,
            +bottom_finder == -union)

        bottom_faces = []
        for face in union.find_faces(bottom_finder):
            bottom_faces.append(BRepComponent(face.brep))

        Hull(Union(*bottom_faces)).create_occurrence(True)

    def test_get_plane(self):
        rect = Rect(1, 1)
        circle = Circle(1)
        circle.place(
            (-circle == +rect) + 2,
            ~circle == ~rect,
            ~circle == ~rect)

        hull = Hull(Union(rect, circle))

        self.assertTrue(hull.get_plane().isCoPlanarTo(rect.get_plane()))
        hull.create_occurrence(True)


def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__],
                                                                pattern="three_circle_intersection",
                                                                )
    unittest.TextTestRunner(failfast=True).run(test_suite)
