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


class Geometry2DTest(test_utils.FscadTestCase):
    def test_rect(self):
        rect(1, 2, name="MyRectangle")

    def test_circle(self):
        circle(1, name="MyCircle")


def run(context):
    test_suite = unittest.defaultTestLoader.loadTestsFromTestCase(Geometry2DTest)
    unittest.TextTestRunner(failfast=True).run(test_suite)
