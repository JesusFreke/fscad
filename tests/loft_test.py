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


class LoftTest(test_utils.FscadTestCase):
    def test_simple_loft(self):
        first = rect(2, 2, name="first")
        second = place(rect(1, 1, name="second"),
                       midAt(atMid(first)), midAt(atMid(first)), midAt(1))
        loft(first, second, name="loft")

    def test_loft_with_hole(self):
        first = rect(2, 2, name="first")
        hole1 = place(circle(.5),
                      midAt(atMid(first)), midAt(atMid(first)), keep())
        diff1 = difference(first, hole1)

        second = place(rect(1, 1, name="second"),
                       midAt(atMid(first)), midAt(atMid(first)), midAt(1))
        hole2 = place(circle(.25, name="hole"),
                     midAt(atMid(first)), midAt(atMid(first)), midAt(1))
        diff2 = difference(second, hole2)
        loft(diff1, diff2, name="loft")

    def test_triple_loft(self):
        first = rect(2, 2, name="first")
        second = place(rect(1, 1, name="second"),
                       midAt(atMid(first)), midAt(atMid(first)), midAt(1))
        third = place(rect(2, 2, name="third"),
                       midAt(atMid(first)), midAt(atMid(first)), midAt(2))
        loft(first, second, third, name="loft")

    def test_rotated_loft(self):
        first = rect(2, 2, name="first")
        second = place(rx(rect(1, 1, name="second"), 90),
                       midAt(atMid(first)), midAt(atMid(first)), midAt(1))
        loft(first, second, name="loft")

    def test_hidden_occurrence_loft(self):
        first = rect(2, 2, name="first")
        second = place(rect(1, 1, name="second"),
                       midAt(atMid(first)), midAt(atMid(first)))
        diff = difference(first, second, name="diff")

        third = place(rect(2, 2, name="third"), midAt(atMid(first)), midAt(atMid(first)), midAt(2))
        fourth = place(rect(1, 1, name="fourth"), midAt(atMid(first)), midAt(atMid(first)), midAt(atMid(third)))
        diff2 = difference(third, fourth, name="diff2")

        loft(second, fourth, name="loft")


from test_utils import load_tests
def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
