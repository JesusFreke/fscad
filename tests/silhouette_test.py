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


class SilhouetteTest(FscadTestCase):
    def test_orthogonal_face_silhouette(self):
        rect = Rect(1, 1)
        silhouette = Silhouette(rect.faces[0], adsk.core.Plane.create(
            Point3D.create(0, 0, -1),
            Vector3D.create(0, 0, 1)))
        silhouette.create_occurrence(True)

        self.assertEquals(silhouette.size().asArray(), rect.size().asArray())

    def test_non_orthogonal_face_silhouette(self):
        rect = Rect(1, 1)
        rect.ry(45)
        silhouette = Silhouette(rect.faces[0], adsk.core.Plane.create(
            Point3D.create(0, 0, -1),
            Vector3D.create(0, 0, 1)))
        silhouette.create_occurrence(True)

        self.assertEquals(silhouette.size().asArray(), (rect.size().x, rect.size().y, 0))

    def test_parallel_face_silhouette(self):
        rect = Rect(1, 1)
        rect.ry(90)
        silhouette = Silhouette(rect.faces[0], adsk.core.Plane.create(
            Point3D.create(0, 0, -1),
            Vector3D.create(0, 0, 1)))
        silhouette.create_occurrence(True)

        self.assertEquals(silhouette.size().asArray(), (0, 0, 0))

    def test_body_silhouette(self):
        box = Box(1, 1, 1)
        box.ry(45)
        silhouette = Silhouette(box.bodies[0], adsk.core.Plane.create(
            Point3D.create(0, 0, -1),
            Vector3D.create(0, 0, 1)))
        silhouette.create_occurrence(True)

        self.assertEquals(silhouette.size().asArray(), (box.size().x, box.size().y, 0))

    def test_component_silhouette(self):
        rect = Rect(1, 1)
        rect.ry(45)
        silhouette = Silhouette(rect, adsk.core.Plane.create(
            Point3D.create(0, 0, -1),
            Vector3D.create(0, 0, 1)))
        silhouette.create_occurrence(True)

        self.assertEquals(silhouette.size().asArray(), (rect.size().x, rect.size().y, 0))

    def test_multiple_disjoint_faces_silhouette(self):
        rect1 = Rect(1, 1)

        rect2 = Rect(1, 1)
        rect2.ry(45)
        rect2.tx(2)

        assembly = Group([rect1, rect2])

        silhouette = Silhouette(assembly.faces, adsk.core.Plane.create(
            Point3D.create(0, 0, -1),
            Vector3D.create(0, 0, 1)))
        silhouette.create_occurrence(True)

        self.assertTrue(abs(silhouette.size().x - assembly.size().x) < app().pointTolerance)
        self.assertTrue(abs(silhouette.size().y - assembly.size().y) < app().pointTolerance)
        self.assertEquals(silhouette.size().z, 0)

    def test_multiple_overlapping_faces_silhouette(self):
        rect1 = Rect(1, 1)

        rect2 = Rect(1, 1)
        rect2.ry(45)
        rect2.translate(.5, .5)

        assembly = Group([rect1, rect2])

        silhouette = Silhouette(assembly.faces, adsk.core.Plane.create(
            Point3D.create(0, 0, -1),
            Vector3D.create(0, 0, 1)))
        silhouette.create_occurrence(True)

        self.assertTrue(abs(silhouette.size().x - assembly.size().x) < app().pointTolerance)
        self.assertTrue(abs(silhouette.size().y - assembly.size().y) < app().pointTolerance)
        self.assertEquals(silhouette.size().z, 0)

    def test_cylinder_silhouette(self):
        cyl = Cylinder(1, 1)
        silhouette = Silhouette(cyl, adsk.core.Plane.create(
            Point3D.create(0, 0, -1),
            Vector3D.create(0, 0, 1)))
        silhouette.create_occurrence(True)

        self.assertEquals(silhouette.size().asArray(), (cyl.size().x, cyl.size().y, 0))

    def test_single_edge(self):
        circle = Circle(1)

        silhouette = Silhouette(circle.edges[0], adsk.core.Plane.create(
            Point3D.create(0, 0, -1),
            Vector3D.create(0, 0, 1)))
        silhouette.create_occurrence(True)

        self.assertEquals(silhouette.size().asArray(), circle.size().asArray())

    def test_multiple_edges(self):
        rect = Rect(1, 1)

        hole1 = Circle(.1)
        hole2 = Circle(.2)

        hole1.place(
            (-hole1 == -rect) + .1,
            (-hole1 == -rect) + .1,
            ~hole1 == ~rect)

        hole2.place(
            (+hole2 == +rect) - .1,
            (+hole2 == +rect) - .1,
            ~hole2 == ~rect)

        assembly = Difference(rect, hole1, hole2)

        silhouette = Silhouette(assembly.faces[0].outer_edges, assembly.get_plane())

        silhouette.create_occurrence(True)

        self.assertEquals(silhouette.size().asArray(), rect.size().asArray())
        self.assertEquals(len(silhouette.edges), 4)

    def test_named_edges(self):
        box = Box(1, 1, 1)
        silhouette = Silhouette(
            box,
            adsk.core.Plane.create(
                Point3D.create(0, 0, -1),
                Vector3D.create(0, 0, 1)),
            named_edges={
                "front": box.shared_edges(box.bottom, box.front),
                "back": box.shared_edges(box.bottom, box.back),
                "left": box.shared_edges(box.bottom, box.left),
                "right": box.shared_edges(box.bottom, box.right)})
        silhouette.create_occurrence(create_children=True)

        edge_finder = Box(.1, .1, .1)
        edge_finder.place(
            ~edge_finder == ~silhouette,
            -edge_finder == -silhouette,
            ~edge_finder == ~silhouette)
        found_edges = silhouette.find_edges(edge_finder)
        named_edges = silhouette.named_edges("front")
        self.assertEquals(len(found_edges), 1)
        self.assertEquals(found_edges, named_edges)

        edge_finder.place(
            ~edge_finder == ~silhouette,
            +edge_finder == +silhouette,
            ~edge_finder == ~silhouette)
        found_edges = silhouette.find_edges(edge_finder)
        named_edges = silhouette.named_edges("back")
        self.assertEquals(len(found_edges), 1)
        self.assertEquals(found_edges, named_edges)

        edge_finder.place(
            +edge_finder == +silhouette,
            ~edge_finder == ~silhouette,
            ~edge_finder == ~silhouette)
        found_edges = silhouette.find_edges(edge_finder)
        named_edges = silhouette.named_edges("right")
        self.assertEquals(len(found_edges), 1)
        self.assertEquals(found_edges, named_edges)

        edge_finder.place(
            -edge_finder == -silhouette,
            ~edge_finder == ~silhouette,
            ~edge_finder == ~silhouette)
        found_edges = silhouette.find_edges(edge_finder)
        named_edges = silhouette.named_edges("left")
        self.assertEquals(len(found_edges), 1)
        self.assertEquals(found_edges, named_edges)



def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__],
                                                                #pattern="named_edges",
                                                                )
    unittest.TextTestRunner(failfast=True).run(test_suite)
