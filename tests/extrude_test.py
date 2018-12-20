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

from fscad import *

import adsk.fusion
import unittest
import test_utils
import importlib
importlib.reload(test_utils)
import test_utils


class ExtrudeTest(test_utils.FscadTestCase):
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

        extrude = ExtrudeTo(rect, sphere.bodies()[0])
        extrude.create_occurrence(True)

    def test_extrude_to_face(self):
        rect = Rect(1, 1)

        sphere = Sphere(5)
        sphere.tz(10)

        extrude = ExtrudeTo(rect, sphere.bodies()[0].faces[0])
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


from test_utils import load_tests
def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
