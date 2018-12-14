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


class SizeTest(test_utils.FscadTestCase):

    def validate_test(self):
        pass

    def test_occurrence_sizeOf(self):
        ball = sphere(1.442)
        self.assertEqual(sizeOf(ball).asArray(), (2*1.442, 2*1.442, 2*1.442))

    def test_list_sizeOf(self):
        ball = sphere(1.442)
        mybox = place(box(5, 5, 5),
                      minAt(atMax(ball)), midAt(atMid(ball)), midAt(atMid(ball)))
        self.assertEqual(sizeOf(ball, mybox).asArray(), (2*1.442+5, 5, 5))


    def test_face_sizeOf(self):
        mybox = box(5, 5, 5)
        self.assertEqual(sizeOf(get_face(mybox, "top")).asArray(), (5, 5, 0))


from test_utils import load_tests
def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
