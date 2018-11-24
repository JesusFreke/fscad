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


class DifferenceTest(test_utils.FscadTestCase):
    def test_simple_difference(self):
        first = box(1, 1, 1, name="first")
        second = translate(box(1, 1, 1, name="second"), x=.5)
        difference(first, second, name="difference")

    def test_disjoint_difference(self):
        first = box(1, 1, 1, name="first")
        second = translate(box(1, 1, 1, name="second"), x=2)
        difference(first, second, name="difference")

    def test_adjoining_difference(self):
        first = box(1, 1, 1, name="first")
        second = translate(box(1, 1, 1, name="second"), x=1)
        difference(first, second, name="difference")

    def test_complete_difference(self):
        first = box(1, 1, 1, name="first")
        second = box(1, 1, 1, name="second")
        difference(first, second, name="difference")

    def test_complex_difference(self):
        first = box(1, 1, 1, name="first")
        second = place(box(.5, 10, 10, name="second"),
                       midAt(atMid(first)), midAt(atMid(first)), midAt(atMid(first)))
        difference1 = difference(first, second, name="difference1")

        third = first = box(1, 1, 1, name="third")
        fourth = place(box(10, 10, .5, name="fourth"),
                       midAt(atMid(first)), midAt(atMid(first)), midAt(atMid(first)))
        difference2 = difference(third, fourth, name="difference2")

        difference3 = difference(difference1, difference2, name="difference3")


def run(context):
    test_suite = unittest.defaultTestLoader.loadTestsFromTestCase(DifferenceTest)
    unittest.TextTestRunner(failfast=True).run(test_suite)
