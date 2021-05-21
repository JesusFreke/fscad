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
import math
import random
import unittest
from . import test_utils
import importlib
importlib.reload(test_utils)
from . import test_utils


class OrientedBoundingBoxTest(test_utils.FscadTestCase):
    def test_oriented_bounding_box(self):
        box = Box(1, 1, 1)
        box.translate(1, 1, 1)

        box.create_occurrence()
        box.oriented_bounding_box(
            Vector3D.create(1, 1, 1),
            Vector3D.create(-1, 1, 0),
            name="test_bounding_box").create_occurrence()

    def test_y_axis_omitted(self):
        random.seed(1)

        box = Box(1, 1, 1)
        box.translate(1, 1, 1)

        x_axis = Vector3D.create(1, 0, 0)
        box.create_occurrence()

        box.oriented_bounding_box(
            x_axis,
            name="test_bounding_box").create_occurrence()



from .test_utils import load_tests
def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__],
                                                                #pattern="positive_offset",
                                                                )
    unittest.TextTestRunner(failfast=True).run(test_suite)
