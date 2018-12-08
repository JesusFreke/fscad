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
import unittest
import test_utils
import importlib
importlib.reload(test_utils)
import test_utils

import fscad
from fscad import *


class ExtrudeTest(test_utils.FscadTestCase):
    def test_simple_extrude(self):
        first = rect(2, 2, name="first")
        extrude(first, 5, name="loft")

    def test_angle_extrude(self):
        first = rect(5, 1, name="first")
        extrude(first, .5, -25, name="loft")

    def test_multiple_face_extrude(self):
        first = rect(2, 2, name="first")
        second = translate(rect(2, 2, name="second"), x=4)

        extrude(union(first, second, name="combined"), 5, name="loft")

    def test_face_with_hole_extrude(self):
        first = rect(2, 2, name="first")
        hole = place(circle(.5, name="hole"),
                       midAt(atMid(first)), midAt(atMid(first)))

        extrude(difference(first, hole, name="face with hole"), 5, name="loft")

    def test_extrude_hidden(self):
        first = rect(2, 2, name="first")
        hole = place(circle(.5, name="hole"),
                     midAt(atMid(first)), midAt(atMid(first)))
        diff = difference(first, hole, name="face with hole")
        hole2 = fscad._duplicate_occurrence(hole, root())
        extrude(hole2, 5, name="hole_extrude")
        for dup in find_all_duplicates(hole):
            self.assertTrue(dup.nativeObject == hole.nativeObject or dup.nativeObject == hole2.nativeObject)
        self.assertEqual(len(find_all_duplicates(hole)), 2)


from test_utils import load_tests
def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
