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

import adsk.core
import adsk.fusion
import fscad
import unittest
import test_utils
import importlib
importlib.reload(test_utils)
import test_utils

from fscad import *
from adsk.core import Vector3D
from adsk.core import Point3D


class FaceTest(test_utils.FscadTestCase):

    def validate_test(self):
        pass

    def test_box_faces(self):
        first = box(1, 1, 1, name="first")
        bottom = get_face(first, "bottom")
        self.assertTrue(bottom.geometry.normal.isParallelTo(Vector3D.create(0, 0, 1)))
        self.assertEqual(bottom.pointOnFace.z, 0)

        top = get_face(first, "top")
        self.assertTrue(top.geometry.normal.isParallelTo(Vector3D.create(0, 0, 1)))
        self.assertEqual(fscad._mm(top.pointOnFace.z), 1)

        right = get_face(first, "right")
        self.assertTrue(right.geometry.normal.isParallelTo(Vector3D.create(1, 0, 0)))
        self.assertEqual(fscad._mm(right.pointOnFace.x), 1)

        left = get_face(first, "left")
        self.assertTrue(left.geometry.normal.isParallelTo(Vector3D.create(1, 0, 0)))
        self.assertEqual(left.pointOnFace.x, 0)

        front = get_face(first, "front")
        self.assertTrue(front.geometry.normal.isParallelTo(Vector3D.create(0, 1, 0)))
        self.assertEqual(front.pointOnFace.y, 0)

        back = get_face(first, "back")
        self.assertTrue(back.geometry.normal.isParallelTo(Vector3D.create(0, 1, 0)))
        self.assertEqual(fscad._mm(back.pointOnFace.y), 1)

    def test_cylinder_faces(self):
        first = cylinder(1, 1, name="first")

        bottom = get_face(first, "bottom")
        self.assertTrue(bottom.geometry.normal.isParallelTo(Vector3D.create(0, 0, 1)))
        self.assertEqual(bottom.pointOnFace.z, 0)

        top = get_face(first, "top")
        self.assertTrue(top.geometry.normal.isParallelTo(Vector3D.create(0, 0, 1)))
        self.assertEqual(fscad._mm(top.pointOnFace.z), 1)

        side = get_face(first, "side")
        self.assertTrue(isinstance(side.geometry, adsk.core.Cylinder))

    def test_sphere_face(self):
        first = sphere(1, name="first")

        surface = get_face(first, "surface")
        self.assertTrue(isinstance(surface.geometry, adsk.core.Sphere))

    def test_duplicated_faces(self):
        first = box(1, 1, 1, name="first")
        duplicate(tx, (0, 2, 4, 6, 8), first)

        faces = get_faces(first.component, "left")
        self.assertEqual(len(faces), 5)

        face_positions = {0, 2, 4, 6, 8}
        for face in faces:
            self.assertTrue(face.geometry.normal.isParallelTo(Vector3D.create(1, 0, 0)))
            self.assertTrue(fscad._mm(face.pointOnFace.x) in face_positions)
            face_positions -= {face.pointOnFace.x}

        face_positions = {0, 2, 4, 6, 8}
        for occurrence in first.childOccurrences:
            face = get_face(occurrence, "left")
            self.assertTrue(fscad._mm(face.pointOnFace.x) in face_positions)
            face_positions -= {face.pointOnFace.x}

    def test_simple_coincident_faces(self):
        first = box(1, 1, 1, name="first")
        second = place(box(1, 1, 1, name="second"),
                       minAt(atMax(first)), midAt(atMid(first)), midAt(atMid(first)))
        second_left = get_face(second, "left")
        first_right = find_coincident_faces(first, second_left)
        self.assertEqual(len(first_right), 1)
        first_right = first_right[0]
        self.assertTrue(first_right.geometry.normal.isParallelTo(Vector3D.create(1, 0, 0)))
        self.assertEqual(fscad._mm(first_right.pointOnFace.x), 1)

    def test_simple_diff_coincident_face(self):
        first = box(1, 1, 1, name="first")
        second = place(box(.5, .5, .5, name="second"),
                       minAt(atMid(first)), midAt(atMid(first)), midAt(atMid(first)))
        diff = difference(first, second, name="diff")

        second_left = get_face(second, "left")
        face = find_coincident_faces(diff, second_left)
        self.assertEqual(len(face), 1)
        face = face[0]
        self.assertTrue(face.geometry.normal.isParallelTo(Vector3D.create(1, 0, 0)))
        self.assertEqual(fscad._mm(face.pointOnFace.x), .5)

    def test_face_on_copied_body(self):
        first = box(1, 1, 1, name="first")
        cut = place(box(.5, .5, 1),
                    maxAt(atMax(first)), midAt(atMid(first)), minAt(atMin(first)))
        diff = difference(first, cut)

        top = get_face(first, "top")
        self.assertTrue(top.geometry.normal.isParallelTo(Vector3D.create(0, 0, 1)))
        self.assertEqual(fscad._mm(top.pointOnFace.z), 1)

        top = get_face(diff, "top")
        self.assertTrue(top.geometry.normal.isParallelTo(Vector3D.create(0, 0, 1)))
        self.assertEqual(fscad._mm(top.pointOnFace.z), 1)


def run(context):
    test_suite = unittest.defaultTestLoader.loadTestsFromTestCase(FaceTest)
    unittest.TextTestRunner(failfast=True).run(test_suite)
