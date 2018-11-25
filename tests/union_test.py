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


class UnionTest(test_utils.FscadTestCase):
    def test_simple_union(self):
        first = box(1, 1, 1, name="first")
        second = translate(box(1, 1, 1, name="second"), x=1)
        union(first, second, name="union")

    def test_overlapping_union(self):
        first = box(1, 1, 1, name="first")
        second = translate(box(1, 1, 1, name="second"), x=.5)
        union(first, second, name="union")

    def test_disjoint_union(self):
        first = box(1, 1, 1, name="first")
        second = translate(box(1, 1, 1, name="second"), x=2)
        union(first, second, name="union")

    def test_single_union(self):
        first = box(1, 1, 1, name="first")
        union(first, name="union")

    def test_overlapping_disjoint_union(self):
        first = box(1, 1, 1, name="first")
        second = translate(box(1, 1, 1, name="second"), x=2)
        first_union = union(first, second, name="first_union")

        third = translate(box(1, 1, 1, name="third"), y=.5)
        fourth = translate(box(1, 1, 1, name="fourth"), x=2, y=.5)
        second_union = union(third, fourth, name="second_union")

        union(first_union, second_union, name="final_union")

    def test_joined_overlapping_disjoint_union(self):
        first = box(1, 1, 1, name="first")
        second = translate(box(1, 1, 1, name="second"), x=2)
        first_union = union(first, second, name="first_union")

        third = translate(box(1, 1, 1, name="third"), y=.5)
        fourth = translate(box(1, 1, 1, name="fourth"), x=2, y=.5)
        second_union = union(third, fourth, name="second_union")

        third_union = union(first_union, second_union, name="third_union")

        fifth = box(3, .1, .1, name="fifth")
        union(fifth, third_union, name="fourth_union")

    def test_simple_sketch_union(self):
        first = rect(1, 1, name="first")
        second = translate(rect(1, 1, name="second"), x=1)
        union(first, second, name="union")

    def test_overlaping_sketch_union(self):
        first = rect(1, 1, name="first")
        second = translate(rect(1, 1, name="second"), x=.5)
        union(first, second, name="union")

    def test_disjoint_sketch_union(self):
        first = rect(1, 1, name="first")
        second = translate(rect(1, 1, name="second"), x=2)
        union(first, second, name="union")

    def test_single_sketch_union(self):
        first = rect(1, 1, name="first")
        union(first, name="union")

    def test_non_coplanar_sketch_union(self):
        first = rotate(rect(1, 1, name="first"), x=45)
        second = place(rotate(rect(1, 1, name="second"), x=45, z=180),
                       minAt(atMin(first)),
                       minAt(atMax(first)),
                       keep())
        got_exception = False
        try:
            union(first, second, name="union")
        except ValueError as ex:
            got_exception = True
        self.assertTrue(got_exception, "No error when unioning non-coplanar sketches")

    def test_empty_sketch_union(self):
        first = rect(1, 1, name="first")
        second = rect(1, 1, name="second")
        empty = difference(first, second, name="empty")

        third = tx(rect(1, 1, name="third"), 5)

        union(third, empty, name="union")

    def test_duplicated_sketch_union(self):
        first = rect(1, 3, name="first")
        firsts = duplicate(tx, (0, 2, 4, 6, 8), first)

        second = ty(rz(rect(1, 9, name="second"), 90), 1)
        seconds = duplicate(ty, (0, 2), second)

        union(firsts, seconds, name="union")

def run(context):
    test_suite = unittest.defaultTestLoader.loadTestsFromTestCase(UnionTest)
    unittest.TextTestRunner(failfast=True).run(test_suite)
