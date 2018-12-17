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


class BasicGeometryTest(test_utils.FscadTestCase):
    def test_box(self):
        box1 = Box(1, 2, 3, "box1")

        self.assertEqual(box1.size().asArray(), (1, 2, 3))
        self.assertEqual(box1.min().asArray(), (0, 0, 0))
        self.assertEqual(box1.mid().asArray(), (.5, 1, 1.5))
        self.assertEqual(box1.max().asArray(), (1, 2, 3))

        bottom = box1.bottom
        self.assertTrue(bottom.geometry.normal.isParallelTo(Vector3D.create(0, 0, 1)))
        self.assertEqual(bottom.pointOnFace.z, 0)

        top = box1.top
        self.assertTrue(top.geometry.normal.isParallelTo(Vector3D.create(0, 0, 1)))
        self.assertEqual(top.pointOnFace.z, 3)

        right = box1.right
        self.assertTrue(right.geometry.normal.isParallelTo(Vector3D.create(1, 0, 0)))
        self.assertEqual(right.pointOnFace.x, 1)

        left = box1.left
        self.assertTrue(left.geometry.normal.isParallelTo(Vector3D.create(1, 0, 0)))
        self.assertEqual(left.pointOnFace.x, 0)

        front = box1.front
        self.assertTrue(front.geometry.normal.isParallelTo(Vector3D.create(0, 1, 0)))
        self.assertEqual(front.pointOnFace.y, 0)

        back = box1.back
        self.assertTrue(back.geometry.normal.isParallelTo(Vector3D.create(0, 1, 0)))
        self.assertEqual(back.pointOnFace.y, 2)

        box1.create_occurrence()

    def test_placed_box_faces(self):
        box1 = Box(1, 2, 3, "box1")
        box2 = Box(1, 1, 1, "box2")
        box1.place(-box1 == +box2,
                   -box1 == +box2,
                   -box1 == +box2)

        bottom = box1.bottom
        self.assertTrue(bottom.geometry.normal.isParallelTo(Vector3D.create(0, 0, 1)))
        self.assertEqual(bottom.pointOnFace.z, 1)

        top = box1.top
        self.assertTrue(top.geometry.normal.isParallelTo(Vector3D.create(0, 0, 1)))
        self.assertEqual(top.pointOnFace.z, 4)

        right = box1.right
        self.assertTrue(right.geometry.normal.isParallelTo(Vector3D.create(1, 0, 0)))
        self.assertEqual(right.pointOnFace.x, 2)

        left = box1.left
        self.assertTrue(left.geometry.normal.isParallelTo(Vector3D.create(1, 0, 0)))
        self.assertEqual(left.pointOnFace.x, 1)

        front = box1.front
        self.assertTrue(front.geometry.normal.isParallelTo(Vector3D.create(0, 1, 0)))
        self.assertEqual(front.pointOnFace.y, 1)

        back = box1.back
        self.assertTrue(back.geometry.normal.isParallelTo(Vector3D.create(0, 1, 0)))
        self.assertEqual(back.pointOnFace.y, 3)

        box1.create_occurrence()


from test_utils import load_tests
def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
