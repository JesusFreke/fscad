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


class PlaceTest(test_utils.FscadTestCase):
    def test_simple_place(self):
        first = box(1, 1, 1, name="first")
        second = place(box(1, 1, 1, name="second"),
                       midAt(atMin(first)),
                       minAt(atMax(first)),
                       maxAt(atMid(first)))

    def test_rotated_place(self):
        first = rz(box(1, 1, 1, name="first"), -45)
        second = place(rz(box(1, 1, 1, name="second"), -45),
                       minAt(atMax(first)), keep(), keep())

    def test_modified_constraint_place(self):
        first = box(1, 1, 1, name="first")
        second = place(box(1, 1, 1, name="second"),
                       midAt(atMid(first)),
                       midAt(atMid(first)),
                       minAt(lambda coord: atMax(first)(coord) + 2))

    def test_simple_place_sketch(self):
        first = rect(1, 1, name="first")
        second = place(rect(1, 1, name="second"),
                       midAt(atMin(first)),
                       minAt(atMax(first)),
                       maxAt(atMid(first)))

    def test_place_tilted_sketch(self):
        first = rotate(rect(1, 1, name="first"), x=45)
        second = place(rotate(rect(1, 1, name="second"), x=45, z=-180),
                       minAt(atMin(first)),
                       minAt(atMax(first)),
                       keep())


from test_utils import load_tests
def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
