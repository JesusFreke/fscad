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


class TransformTest(FscadTestCase):
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

    def test_rxryrz(self):
        box1 = Box(1, 2, 3, "box1")
        box1.rx(30).ry(30).rz(30)
        box1.create_occurrence()

    def test_rx_with_center(self):
        box1 = Box(1, 2, 3, "box1")
        center = Point3D.create(1, 2, 3)
        box1.rx(30, center)
        box1.create_occurrence()

    def test_translation(self):
        box1 = Box(1, 2, 3, "box1")
        box1.translate(1, 2, 3)
        box1.create_occurrence()

        self.assertEqual(box1.size().asArray(), (1, 2, 3))
        self.assertEqual(box1.min().asArray(), (1, 2, 3))
        self.assertEqual(box1.mid().asArray(), (1.5, 3, 4.5))
        self.assertEqual(box1.max().asArray(), (2, 4, 6))

    def test_txtytz(self):
        box1 = Box(1, 2, 3, "box1")
        box1.tx(1).ty(2).tz(3)
        box1.create_occurrence()

        self.assertEqual(box1.size().asArray(), (1, 2, 3))
        self.assertEqual(box1.min().asArray(), (1, 2, 3))
        self.assertEqual(box1.mid().asArray(), (1.5, 3, 4.5))
        self.assertEqual(box1.max().asArray(), (2, 4, 6))

    def test_basic_uniform_scale(self):
        box = Box(1, 2, 3)
        box.scale(2, 2, 2)
        box.create_occurrence()

        self.assertEqual(box.size().asArray(), (2, 4, 6))
        self.assertEqual(box.min().asArray(), (0, 0, 0))
        self.assertEqual(box.mid().asArray(), (1, 2, 3))
        self.assertEqual(box.max().asArray(), (2, 4, 6))

    def test_non_uniform_scale(self):
        box = Box(1, 2, 3)

        try:
            box.scale(2, 1, 1)
            self.fail("Expected error did not occur")
        except ValueError:
            pass

    def test_uniform_scale_with_center(self):
        box = Box(1, 2, 3)
        box.scale(2, 2, 2, (1, 2, 3))
        box.create_occurrence()

        self.assertEqual(box.size().asArray(), (2, 4, 6))
        self.assertEqual(box.min().asArray(), (-1, -2, -3))
        self.assertEqual(box.mid().asArray(), (0, 0, 0))
        self.assertEqual(box.max().asArray(), (1, 2, 3))

    def test_basic_mirror(self):
        box = Box(1, 2, 3)
        box.scale(-1, 1, 1)
        box.create_occurrence()

        self.assertEqual(box.size().asArray(), (1, 2, 3))
        self.assertEqual(box.min().asArray(), (-1, 0, 0))
        self.assertEqual(box.mid().asArray(), (-.5, 1, 1.5))
        self.assertEqual(box.max().asArray(), (0, 2, 3))

    def test_mirror_with_center(self):
        box = Box(1, 2, 3)
        box.scale(-1, 1, 1, (1, 0, 0))
        box.create_occurrence()

        self.assertEqual(box.size().asArray(), (1, 2, 3))
        self.assertEqual(box.min().asArray(), (1, 0, 0))
        self.assertEqual(box.mid().asArray(), (1.5, 1, 1.5))
        self.assertEqual(box.max().asArray(), (2, 2, 3))

    def test_mixed_scale_mirror(self):
        box = Box(1, 2, 3)
        box.scale(-2, 2, 2, (1, 0, 0))
        box.create_occurrence()

        self.assertEqual(box.size().asArray(), (2, 4, 6))
        self.assertEqual(box.min().asArray(), (1, 0, 0))
        self.assertEqual(box.mid().asArray(), (2, 2, 3))
        self.assertEqual(box.max().asArray(), (3, 4, 6))

    def test_transform(self):
        box = Box(1, 2, 3)

        matrix = adsk.core.Matrix3D.create()
        matrix.setToRotateTo(
            adsk.core.Vector3D.create(0, 0, 1),
            adsk.core.Vector3D.create(1, 1, 1))
        matrix.translation = adsk.core.Vector3D.create(1, 2, 3)

        box.transform(matrix)
        box.create_occurrence()

    def test_world_transform(self):
        box = Box(1, 2, 3, name="box")
        box.translate(1, 2, 3)

        self.assertEqual(box.world_transform().asArray(),
                         (1.0, 0.0, 0.0, 1.0,
                          0.0, 1.0, 0.0, 2.0,
                          0.0, 0.0, 1.0, 3.0,
                          0.0, 0.0, 0.0, 1.0))

        reverse_transform = box.world_transform().copy()
        reverse_transform.invert()
        box.transform(reverse_transform)
        box.create_occurrence()


def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
