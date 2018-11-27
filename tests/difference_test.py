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

    def test_duplicated_difference(self):
        first = box(1, 3, 1, name="first")
        firsts = duplicate(tx, (0, 2, 4, 6, 8), first)

        second = ty(rz(box(1, 9, 1, name="second"), -90), 1)
        seconds = duplicate(ty, (0, 2), second)

        difference(firsts, seconds, name="difference")

    def test_empty_difference(self):
        first = box(1, 1, 1, name="first")
        second = box(1, 1, 1, name="second")
        difference1 = difference(first, second, name="difference1")
        third = box(1, 1, 1, name="third")
        difference2 = difference(difference1, third, name="difference2")

    def test_simple_sketch_difference(self):
        first = rect(1, 1, name="first")
        second = tx(rect(1, 1, name="second"), .5)
        difference(first, second, name="difference")

    def test_disjoint_sketch_difference(self):
        first = rect(1, 1, name="first")
        second = tx(rect(1, 1, name="second"), 2)
        difference(first, second, name="difference")

    def test_adjoining_sketch_difference(self):
        first = rect(1, 1, name="first")
        second = tx(rect(1, 1, name="second"), 1)
        difference(first, second, name="difference")

    def test_complete_sketch_difference(self):
        first = rect(1, 1, name="first")
        second = rect(1, 1, name="second")
        difference(first, second, name="difference")

    def test_empty_sketch_difference(self):
        first = rect(1, 1, name="first")
        second = rect(1, 1, name="second")
        difference1 = difference(first, second, name="difference1")
        third = rect(1, 1, name="third")
        difference2 = difference(difference1, third, name="difference2")

    def test_complex_sketch_difference(self):
        first = rect(1, 1, name="first")
        second = tx(rect(.5, 1, name="second"), .25)
        difference1 = difference(first, second, name="difference1")

        third = rect(1, 1, name="third")
        fourth = ty(rect(1, .5, name="fourth"), .25)
        difference2 = difference(third, fourth, name="difference2")

        difference3 = difference(difference1, difference2, name="difference3")

    def test_duplicated_sketch_difference(self):
        first = rect(1, 3, name="first")
        firsts = duplicate(tx, (0, 2, 4, 6, 8), first)

        second = ty(rz(rect(1, 9, name="second"), -90), 1)
        seconds = duplicate(ty, (0, 2), second)

        difference(firsts, seconds, name="difference")

    def test_sketch_body_difference(self):
        first = rect(1, 1, name="first")
        second = translate(box(1, 1, 1, name="second"), x=.5)
        difference(first, second, name="difference")

    def test_body_sketch_difference(self):
        first = rect(1, 1, name="first")
        second = translate(box(1, 1, 1, name="second"), x=.5)
        got_exception = False
        try:
            difference(second, first, name="difference")
        except ValueError:
            got_exception = True
        self.assertTrue(got_exception, "No error when subtracting a sketch from a body")

    def test_inside_hole_sketch_difference(self):
        outer = rect(1, 1, name="outer")
        inner = place(rect(.5, .5, name="inner"),
                      midAt(atMid(outer)), midAt(atMid(outer)), keep())
        difference(outer, inner)

    def test_non_coplanar_sketch_difference(self):
        first = rotate(rect(1, 1, name="first"), x=45)
        second = place(rotate(rect(1, 1, name="second"), x=45, z=180),
                       minAt(atMin(first)),
                       minAt(atMax(first)),
                       keep())
        got_exception = False
        try:
            difference(first, second, name="difference")
        except ValueError:
            got_exception = True
        self.assertTrue(got_exception, "No error when subtracting non-coplanar sketches")

    def test_difference_with_inside_hole(self):
        outer = rect(1, 1, name="outer")
        inner = place(rect(.5, .5, name="inner"),
                      midAt(atMid(outer)), midAt(atMid(outer)), keep())
        diff1 = difference(outer, inner, name="difference1")

        outerouter = place(rect(2, 2, name="outerouter"),
                           midAt(atMid(outer)), midAt(atMid(outer)))
        difference(outerouter, diff1, name="difference2")


def run(context):
    #test_suite = test_suite = unittest.defaultTestLoader.loadTestsFromName(
    #    "difference_test.DifferenceTest.test_complex_sketch_difference")

    test_suite = unittest.defaultTestLoader.loadTestsFromTestCase(DifferenceTest)
    unittest.TextTestRunner(failfast=True).run(test_suite)
