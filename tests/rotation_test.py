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


class RotationTest(test_utils.FscadTestCase):
    def test_simple_x_rotation(self):
        box1 = Box(1, 2, 3, "box1")
        box1.rotate(30)
        box1.create_occurrence()

    def test_simple_y_rotation(self):
        box1 = Box(1, 2, 3, "box1")
        box1.rotate(ry=30)
        box1.create_occurrence()

    def test_simple_z_rotation(self):
        box1 = Box(1, 2, 3, "box1")
        box1.rotate(rz=30)
        box1.create_occurrence()

    def test_compound_rotation(self):
        box1 = Box(1, 2, 3, "box1")
        box1.rotate(30, 30, 30)
        box1.create_occurrence()

    def test_x_rotation_with_center(self):
        box1 = Box(1, 2, 3, "box1")
        box1.rotate(30, center=(1, 2, 3))
        box1.create_occurrence()

    def test_compound_rotation_with_center(self):
        box1 = Box(1, 2, 3, "box1")
        box1.rotate(30, 30, 30, center=(1, 2, 3))
        box1.create_occurrence()

    def test_double_rotation(self):
        box1 = Box(1, 2, 3, "box1")
        box1.rotate(rz=45)
        box1.rotate(rz=45)
        box1.create_occurrence()


from test_utils import load_tests
def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
