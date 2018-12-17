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


class DifferenceTest(test_utils.FscadTestCase):
    def test_basic_difference(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(.5, .5, .5, "box2")
        box2.place(+box2 == +box1,
                   ~box2 == ~box1,
                   ~box2 == ~box1)
        Difference(box1, box2).create_occurrence(True)

    def test_disjoint_difference(self):
        box1 = Box(1, 1, 1, name="box1")
        box2 = Box(1, 1, 1, name="box2")
        box2.place((-box2 == +box1) + 1,
                   ~box2 == ~box1,
                   ~box2 == ~box1)
        Difference(box1, box2).create_occurrence(True)

    def test_adjoining_difference(self):
        box1 = Box(1, 1, 1, name="box1")
        box2 = Box(1, 1, 1, name="box2")
        box2.place((-box2 == +box1),
                   ~box2 == ~box1,
                   ~box2 == ~box1)
        Difference(box1, box2).create_occurrence(True)

    def test_complete_difference(self):
        box1 = Box(1, 1, 1, name="box1")
        box2 = Box(1, 1, 1, name="box2")
        Difference(box1, box2).create_occurrence(True)

    def test_complex_difference(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(.5, 10, 10, "box2")
        box2.place(~box2 == ~box1,
                   ~box2 == ~box1,
                   ~box2 == ~box1)
        difference1 = Difference(box1, box2, name="difference1")

        box3 = Box(1, 1, 1, "box3")
        box4 = Box(10, 10, .5, "box4")
        box4.place(~box4 == ~box1,
                   ~box4 == ~box1,
                   ~box4 == ~box1)

        difference2 = Difference(box3, box4, name="difference2")
        Difference(difference1, difference2, name="difference3").create_occurrence(True)

    def test_empty_difference(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        difference1 = Difference(box1, box2, name="difference1")
        box3 = Box(1, 1, 1, "box3")
        Difference(difference1, box3, name="difference2").create_occurrence(True)


from test_utils import load_tests
def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
