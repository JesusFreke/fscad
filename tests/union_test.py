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

    def test_duplicated_union(self):
        first = box(1, 3, 1, name="first")
        firsts = duplicate(tx, (0, 2, 4, 6, 8), first)

        #union(*firsts.childOccurrences)

        second = ty(rz(box(1, 9, 1, name="second"), -90), 1)
        seconds = duplicate(ty, (0, 2), second)

        union(firsts, seconds, name="union")

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

        second = ty(rz(rect(1, 9, name="second"), -90), 1)
        seconds = duplicate(ty, (0, 2), second)

        union(firsts, seconds, name="union")

    def test_hidden_base_occurrence_union(self):
        first = box(1, 1, 1, name="first")
        second = place(box(2, 2, 2, name="second"),
                       minAt(atMid(first)), midAt(atMid(first)), midAt(atMid(first)))
        result = difference(first, second, name="difference")

        third = place(box(2, 2, 2, name="third"),
                      minAt(atMax(second)), midAt(atMid(second)), midAt(atMid(second)))

        got_error = False
        try:
            union(second, third, name="second_union")
        except:
            got_error = True
        self.assertTrue(got_error)

    def test_hidden_tool_occurrence_union(self):
        first = box(1, 1, 1, name="first")
        second = place(box(2, 2, 2, name="second"),
                       minAt(atMid(first)), midAt(atMid(first)), midAt(atMid(first)))
        result = difference(first, second, name="difference")

        third = place(box(2, 2, 2, name="third"),
                      minAt(atMax(second)), midAt(atMid(second)), midAt(atMid(second)))
        union(third, second, name="second_union")

    def test_keep_base(self):
        with keep_subtree(False):
            first = box(1, 1, 1, name="first")
            keep_bodies(first)
            second = place(box(.5, .5, .5, name="second"),
                           minAt(atMax(first)), midAt(atMid(first)), midAt(atMid(first)))
            union(first, second, name="union")

    def test_keep_base_recursive(self):
        with keep_subtree(False):
            first = box(1, 1, 1, name="first")
            keep_bodies(first)
            second = place(box(.5, .5, .5, name="second"),
                           minAt(atMax(first)), midAt(atMid(first)), midAt(atMid(first)))
            union1 = union(first, second, name="union1")

            third = place(box(.5, .5, .5, name="third"),
                          midAt(atMid(first)), maxAt(atMin(first)), midAt(atMid(first)))
            union(union1, third, name="union2")

    def test_keep_tool(self):
        with keep_subtree(False):
            first = box(1, 1, 1, name="first")
            second = place(box(.5, .5, .5, name="second"),
                           minAt(atMax(first)), midAt(atMid(first)), midAt(atMid(first)))
            keep_bodies(second)
            union(first, second, name="union")

    def test_keep_tool_recursive(self):
        with keep_subtree(False):
            first = box(1, 1, 1, name="first")
            second = place(box(.5, .5, .5, name="second"),
                           minAt(atMax(first)), midAt(atMid(first)), midAt(atMid(first)))
            keep_bodies(second)
            union1 = union(first, second, name="union1")

            third = place(box(.5, .5, .5, name="third"),
                          midAt(atMid(second)), maxAt(atMin(second)), midAt(atMid(second)))
            union(union1, third, name="union2")

    def test_deep_keep(self):
        set_parametric(True)
        first = box(1, 1, 1, name="first")
        second = place(box(.5, .5, .5, name="second"),
                       minAt(atMax(first)), midAt(atMid(first)), midAt(atMid(first)))
        keep_bodies(first)
        union1 = union(first, second, name="union1")
        with keep_subtree(False):
            third = place(box(.5, .5, .5, name="third"),
                          midAt(atMid(first)), maxAt(atMin(first)), midAt(atMid(first)))
            union2 = union(union1, third, name="union2")

    def test_keep_subtree(self):
        set_parametric(True)
        first = box(1, 1, 1, name="first")
        second = place(box(.5, .5, .5, name="second"),
                       minAt(atMax(first)), midAt(atMid(first)), midAt(atMid(first)))
        union1 = union(first, second, name="union1")
        keep_bodies(union1)
        with keep_subtree(False):
            third = place(box(.5, .5, .5, name="third"),
                          midAt(atMid(first)), maxAt(atMin(first)), midAt(atMid(first)))
            union2 = union(union1, third, name="union2")


    def test_keep_duplicated_tool(self):
        with keep_subtree(False):
            first = box(1, 1, 1, name="first")
            second = place(box(.5, .5, .5, name="second"),
                           minAt(atMax(first)), midAt(atMid(first)), midAt(atMid(first)))
            keep_bodies(second)
            dup = duplicate(lambda o, v: rz(o, v, center=midOf(first).asArray()), (0, -90), second)

            union1 = union(first, dup, name="union1")

            third = place(box(.25, .25, .25, name="third"),
                          minAt(atMax(second)), midAt(atMid(second)), midAt(atMid(second)))
            union(union1, third, name="union2")

    def test_keep_duplicated_target(self):
        with keep_subtree(False):
            first = box(1, 1, 1, name="first")
            keep_bodies(first)
            dup = duplicate(tx, (0, 2, 4, 6, 8), first)

            second = place(box(10, .5, .5),
                           minAt(atMin(first)), midAt(atMid(first)), midAt(atMid(first)))

            union1 = union(dup, second, name="union")

    def test_collapse_sequential_unions(self):
        first = box(1, 1, 1, name="first")
        second = place(box(1, 1, 1, name="second"),
                       minAt(atMax(first)), midAt(atMid(first)), midAt(atMid(first)))
        first = union(first, second)
        third = place(box(1, 1, 1, name="third"),
                      minAt(atMax(second)), midAt(atMid(second)), midAt(atMid(second)))
        union(first, third)


from test_utils import load_tests
def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]
                                                                #, pattern="keep_duplicated_target"
                                                                )
    unittest.TextTestRunner(failfast=True).run(test_suite)
