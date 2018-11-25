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

from fscad import *


class RotateTest(test_utils.FscadTestCase):
    def test_rotate_x(self):
        rotate(box(1, 2, 3), x=45)

    def test_rotate_y(self):
        rotate(box(1, 2, 3), y=45)

    def test_rotate_z(self):
        rotate(box(1, 2, 3), z=45)

    def test_rotate_all(self):
        rotate(box(1, 2, 3), x=10, y=20, z=30)

    def test_rotate_all_sequentially(self):
        r1 = rotate(box(1, 2, 3), z=30)
        r2 = rotate(r1, y=20)
        r3 = rotate(r2, x=10)

    def test_rx(self):
        rx(box(1, 2, 3), 45)

    def test_ry(self):
        ry(box(1, 2, 3), 45)

    def test_rz(self):
        rz(box(1, 2, 3), 45)

    def test_rxryrz(self):
        rz(ry(rx(box(1, 2, 3), 10), 20), 30)

    def test_rotate_translated(self):
        rz(tx(box(1, 2, 3), 10), 45)


def run(context):
    test_suite = unittest.defaultTestLoader.loadTestsFromTestCase(RotateTest)
    unittest.TextTestRunner(failfast=True).run(test_suite)
