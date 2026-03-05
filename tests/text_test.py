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

import unittest

from adsk.core import Vector3D

# note: load_tests is required for the "pattern" test filtering functionality in loadTestsFromModule in run()
from fscad.test_utils import FscadTestCase, load_tests
from fscad.fscad import *


class TextTest(FscadTestCase):
    def test_text(self):
        text = Text("A", 1.0)

        self.assertGreater(len(text.bodies), 0)
        self.assertGreater(text.size().x, 0)
        self.assertGreater(text.size().y, 0)

        text.create_occurrence(True)

    def test_text_with_holes(self):
        text = Text("8", 1.0)

        self.assertEqual(len(text.bodies), 1)

        text.create_occurrence(True)

    def test_text_multiple_chars(self):
        text = Text("AB", 1.0)

        self.assertGreater(len(text.bodies), 1)

        text.create_occurrence(True)

    def test_text_get_plane(self):
        text = Text("A", 1.0)

        plane = text.get_plane()
        self.assertIsNotNone(plane)
        self.assertTrue(plane.normal.isParallelTo(Vector3D.create(0, 0, 1)))

        text.create_occurrence(True)

    def test_text_extrude(self):
        text = Text("A", 1.0)
        extruded = Extrude(text, 1.0)

        self.assertGreater(extruded.size().z, 0)

        extruded.create_occurrence(True)

    def test_text_font(self):
        text = Text("A", 1.0, font="Arial")

        self.assertGreater(len(text.bodies), 0)

        text.create_occurrence(True)

    def test_text_empty(self):
        text = Text(" ", 1.0)

        self.assertEqual(len(text.bodies), 0)


def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(
        sys.modules[__name__],
    )
    unittest.TextTestRunner(failfast=True).run(test_suite)
