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


class CopyTest(test_utils.FscadTestCase):
    def test_box_copy(self):
        box = Box(1, 1, 1)
        box_copy = box.copy()
        box_copy.place(
            -box_copy == +box,
            ~box_copy == ~box,
            ~box_copy == ~box)

        box.create_occurrence()
        box_copy.create_occurrence()

    def test_union_copy(self):
        box = Box(1, 1, 1)
        box2 = Box(1, 1, 1, "Box2")
        box2.place(-box2 == +box,
                   ~box2 == ~box,
                   ~box2 == ~box)
        union = Union(box, box2)

        box3 = Box(1, 1, 1, "Box3")
        box3.place(-box3 == +box2,
                   ~box3 == ~box2,
                   ~box3 == ~box2)

        union2 = Union(union, box3, name="Union2")

        union2_copy = union2.copy()
        union2_copy.place((-union2_copy == +union2) + 1,
                          ~union2_copy == ~union2,
                          ~union2_copy == ~union2)

        union2.create_occurrence(True)
        union2_copy.create_occurrence(True)


from test_utils import load_tests
def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
