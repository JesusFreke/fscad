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


class ExtrudeTest(FscadTestCase):
    def test_basic_extrude(self):
        rect = Rect(1, 1)

        extrude = Extrude(rect, 1)
        extrude.create_occurrence(True)

        self.assertEqual(len(extrude.start_faces), 1)
        self.assertEqual(extrude.start_faces[0].brep.pointOnFace.z, 0)
        self.assertEqual(len(extrude.end_faces), 1)
        self.assertEqual(extrude.end_faces[0].brep.pointOnFace.z, 1)
        self.assertEqual(len(extrude.side_faces), 4)

    def test_two_face_extrude(self):
        rect = Rect(2, 1, "rect")
        splitter = Rect(1, 1, "splitter")
        splitter.place(~splitter == ~rect)
        split = Difference(rect, splitter, name="split")
        extrude = Extrude(split, 1)
        extrude.create_occurrence(True)
        self.assertEqual(len(extrude.start_faces), 2)
        self.assertEqual(extrude.start_faces[0].brep.pointOnFace.z, 0)
        self.assertEqual(extrude.start_faces[1].brep.pointOnFace.z, 0)
        self.assertEqual(len(extrude.end_faces), 2)
        self.assertEqual(extrude.end_faces[0].brep.pointOnFace.z, 1)
        self.assertEqual(extrude.end_faces[1].brep.pointOnFace.z, 1)
        self.assertEqual(len(extrude.side_faces), 8)

    def test_extrude_to(self):
        rect = Rect(1, 1)

        sphere = Sphere(5)
        sphere.tz(10)

        extrude = ExtrudeTo(rect, sphere)
        extrude.create_occurrence(True)

    def test_extrude_to_body(self):
        rect = Rect(1, 1)

        sphere = Sphere(5)
        sphere.tz(10)

        extrude = ExtrudeTo(rect, sphere.bodies[0])
        extrude.create_occurrence(True)

    def test_extrude_to_face(self):
        rect = Rect(1, 1)

        sphere = Sphere(5)
        sphere.tz(10)

        extrude = ExtrudeTo(rect, sphere.bodies[0].faces[0])
        extrude.create_occurrence(True)

    def test_extrude_face(self):
        box = Box(1, 1, 1)
        extrude = Extrude(box.right, 1)
        extrude.create_occurrence(True)

    def test_extrude_face_to(self):
        box = Box(1, 1, 1)

        sphere = Sphere(5)
        sphere.tz(10)

        extrude = ExtrudeTo(box.top, sphere)
        extrude.create_occurrence(True)

    def test_multiple_split_face_extrude(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        box2.place(-box2 == +box1)
        circle1 = Circle(.5, "circle1")
        circle1.place(~circle1 == ~box1,
                      ~circle1 == ~box1,
                      ~circle1 == +box1)
        circle2 = Circle(.5, "circle2")
        circle2.place(~circle2 == ~box2,
                      ~circle2 == ~box2,
                      ~circle2 == +box2)
        box = Union(box1, box2)
        split = SplitFace(box, Union(circle1, circle2))
        extrude = Extrude(split.find_faces((circle1, circle2)), 1)
        extrude.create_occurrence(True)

    def test_multiple_split_face_extrude_to(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        box2.place(-box2 == +box1)
        circle1 = Circle(.5, "circle1")
        circle1.place(~circle1 == ~box1,
                      ~circle1 == ~box1,
                      ~circle1 == +box1)
        circle2 = Circle(.5, "circle2")
        circle2.place(~circle2 == ~box2,
                      ~circle2 == ~box2,
                      ~circle2 == +box2)
        box = Union(box1, box2)
        split = SplitFace(box, Union(circle1, circle2))

        sphere = Sphere(2)
        sphere.place(~sphere == ~box, ~sphere == ~box, (-sphere == +box) + 2)

        extrude = ExtrudeTo(split.find_faces((circle1, circle2)), sphere)
        extrude.create_occurrence(True)


    def build_relief_body(self, relief_cylinder: Cylinder, extrude_direction: Vector3D):
        extrude_vector = extrude_direction
        extrude_vector.scaleBy(relief_cylinder.radius * 2)

        second_cylinder = relief_cylinder.copy().translate(*extrude_vector.asArray())

        cut_rect = Rect(
            max(relief_cylinder.radius * 2, relief_cylinder.height) * 2,
            max(relief_cylinder.radius * 2, relief_cylinder.height) * 2)
        cut_rect_transform = Matrix3D.create()
        cut_rect_transform.setToRotateTo(
            cut_rect.get_plane().normal,
            extrude_direction)
        cut_rect.transform(cut_rect_transform)
        cut_rect.place(
            ~cut_rect == ~relief_cylinder,
            ~cut_rect == ~relief_cylinder,
            ~cut_rect == ~relief_cylinder)

        rect = Intersection(cut_rect, relief_cylinder)
        box = Extrude(rect, extrude_vector.length)

        full_shape = Union(relief_cylinder, second_cylinder, box)
        bottom = BRepComponent(full_shape.find_faces(relief_cylinder.top)[0].brep)

        half_shape = Extrude(bottom, 10, name="relief_body")
        #half_shape_translation = bottom.get_plane().normal
        #half_shape_translation.scaleBy(-.02)
        #half_shape.translate(*half_shape_translation.asArray())

        return half_shape

    def test_extrude_extruded_face(self):
        rect = Rect(2, 2, name="rect")
        box = Extrude(rect, 1, name="extruded_box")
        # for some reason, this causes the Extrusion feature to include the face in the feature bodies
        extrusion = Extrude(BRepComponent(box.find_faces(rect)[0].brep), 10, name="re-extrusion")

        # make sure that only the solid extruded body is part of the Extrude component. The face body should be
        # filtered out
        self.assertEqual(len(extrusion.bodies), 1)
        self.assertTrue(extrusion.bodies[0].brep.isSolid)
        extrusion.create_occurrence(create_children=True)

    def test_extrude_to_with_offset(self):
        rect = Rect(1, 1)

        sphere = Sphere(5)
        sphere.tz(10)

        extrude = ExtrudeTo(rect, sphere.bodies[0], offset=1)
        extrude.create_occurrence(True)

    def test_extrude_to_with_negative_offset(self):
        rect = Rect(1, 1)

        sphere = Sphere(5)
        sphere.tz(10)

        extrude = ExtrudeTo(rect, sphere.bodies[0], offset=-1)
        extrude.create_occurrence(True)


def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
