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


class FaceTest(test_utils.FscadTestCase):
    def validate_test(self):
        pass

    def test_basic_shared_edges(self):
        box1 = Box(1, 1, 1, "box1")
        edges = box1.shared_edges(box1.left, box1.front)

        self.assertEqual(len(edges), 1)
        self.assertTrue(isinstance(edges[0].brep.geometry, adsk.core.Line3D))
        self.assertEqual(edges[0].brep.geometry.startPoint.asArray(), (0, 0, 1))
        self.assertEqual(edges[0].brep.geometry.endPoint.asArray(), (0, 0, 0))

    def test_no_shared_edges(self):
        box1 = Box(1, 1, 1, "box1")
        edges = box1.shared_edges(box1.left, box1.right)
        self.assertEqual(len(edges), 0)

    def test_multiple_shared_edges(self):
        box1 = Box(1, 1, 1, "box1")
        sphere = Sphere(.25)
        sphere.place(~sphere == +box1,
                     ~sphere == ~box1,
                     ~sphere == +box1)
        diff = Difference(box1, sphere)
        edges = diff.shared_edges(box1.top, box1.right)
        self.assertEqual(len(edges), 2)

    def test_multiple_faces(self):
        box1 = Box(1, 1, 1, "box1")
        edges = box1.shared_edges([box1.top, box1.bottom], [box1.right, box1.front])
        self.assertEqual(len(edges), 4)

    def test_multiple_faces_both(self):
        box1 = Box(1, 1, 1, "box1")
        edges = box1.shared_edges([box1.top, box1.bottom, box1.right, box1.front],
                                  [box1.top, box1.bottom, box1.right, box1.front])
        self.assertEqual(len(edges), 5)


from test_utils import load_tests
def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
