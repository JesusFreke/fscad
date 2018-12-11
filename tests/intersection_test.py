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


class IntersectionTest(test_utils.FscadTestCase):
    def test_simple_intersection(self):
        first = box(1, 1, 1, name="first")
        second = translate(box(1, 1, 1, name="second"), x=.5)
        intersection(first, second, name="intersection")

    def test_disjoint_intersection(self):
        first = box(1, 1, 1, name="first")
        second = translate(box(1, 1, 1, name="second"), x=2)
        intersection(first, second, name="intersection")

    def test_adjoining_intersection(self):
        first = box(1, 1, 1, name="first")
        second = translate(box(1, 1, 1, name="second"), x=1)
        intersection(first, second, name="intersection")

    def test_complete_intersection(self):
        first = box(1, 1, 1, name="first")
        second = box(1, 1, 1, name="second")
        intersection(first, second, name="intersection")

    def test_complex_intersection(self):
        first = box(1, 1, 1, name="first")
        second = place(box(.5, 10, 10, name="second"),
                       midAt(atMid(first)), midAt(atMid(first)), midAt(atMid(first)))
        difference1 = difference(first, second, name="difference1")

        third = first = box(1, 1, 1, name="third")
        fourth = place(box(10, 10, .5, name="fourth"),
                       midAt(atMid(first)), midAt(atMid(first)), midAt(atMid(first)))
        difference2 = difference(third, fourth, name="difference2")

        intersection1 = intersection(difference1, difference2, name="intersection")

    def test_intersection_with_empty(self):
        first = box(1, 1, 1, name="first")
        second = translate(box(1, 1, 1, name="second"), x=10)
        empty_intersection = intersection(first, second, name="empty_intersection")
        third = box(1, 1, 1, name="third")
        intersection(third, empty_intersection, name="intersection")

    def test_basic_sketch_intersection(self):
        first = rect(1, 1, name="first")
        second = translate(rect(1, 1, name="second"), x=.5)
        intersection(first, second, name="intersection")

    def test_disjoint_sketch_intersection(self):
        first = rect(1, 1, name="first")
        second = translate(rect(1, 1, name="second"), x=2)
        intersection(first, second, name="intersection")

    def test_adjoining_sketch_intersection(self):
        first = rect(1, 1, name="first")
        second = translate(rect(1, 1, name="second"), x=1)
        intersection(first, second, name="intersection")

    def test_complete_sketch_intersection(self):
        first = rect(1, 1, name="first")
        second = rect(1, 1, name="second")
        intersection(first, second, name="intersection")

    def test_complex_sketch_intersection(self):
        first = rect(1, 3, name="first")
        second = translate(rect(1, 3, name="second"), x=2)
        union1 = union(first, second, name="union1")
        third = ty(rz(rect(1, 3, name="third"), -90), 1)
        fourth = ty(rz(rect(1, 3, name="fourth"), -90), 3)
        union2 = union(third, fourth, name="union2")
        intersection(union1, union2, name="intersection")

    def test_duplicated_sketch_intersection(self):
        first = rect(1, 3, name="first")
        firsts = duplicate(tx, (0, 2, 4, 6, 8), first)

        second = ty(rz(rect(1, 9, name="second"), -90), 1)
        seconds = duplicate(ty, (0, 2), second)

        intersection(firsts, seconds, name="intersection")

    def test_sketch_body_intersection(self):
        first = rect(1, 1, name="first")
        second = translate(box(1, 1, 1, name="second"), x=.5)
        intersection(first, second, name="intersection")

    def test_body_sketch_intersection(self):
        first = rect(1, 1, name="first")
        second = translate(box(1, 1, 1, name="second"), x=.5)
        result = intersection(second, first, name="intersection")

    def test_sketch_intersection_with_empty(self):
        first = rect(1, 1, name="first")
        second = translate(rect(1, 1, name="second"), x=10)
        empty_intersection = intersection(first, second, name="empty_intersection")
        third = rect(1, 1, name="third")
        intersection(third, empty_intersection, name="intersection")

    def test_non_coplanar_sketch_intersection(self):
        first = rotate(rect(1, 1, name="first"), x=45)
        second = place(rotate(rect(1, 1, name="second"), x=45, z=180),
                       minAt(atMin(first)),
                       minAt(atMax(first)),
                       keep())
        got_exception = False
        try:
            intersection(first, second, name="intersection")
        except ValueError as ex:
            got_exception = True
        self.assertTrue(got_exception, "No error when intersecting non-coplanar sketches")

    def test_hidden_base_occurrence_intersection(self):
        first = box(1, 1, 1, name="first")
        second = place(box(2, 2, 2, name="second"),
                       minAt(atMid(first)), midAt(atMid(first)), midAt(atMid(first)))
        result = intersection(first, second, name="first_intersection")

        got_error = False
        try:
            intersection(second, result, name="second_intersection")
        except:
            got_error = True
        self.assertTrue(got_error)

    def test_hidden_tool_occurrence_intersection(self):
        first = box(1, 1, 1, name="first")
        second = place(box(2, 2, 2, name="second"),
                       minAt(atMid(first)), midAt(atMid(first)), midAt(atMid(first)))
        result = intersection(first, second, name="first_intersection")
        intersection(result, second, name="second_intersection")
        self.assertEqual(len(find_all_duplicates(second)), 2)

    def test_keep_base(self):
        with keep_subtree(False):
            first = box(1, 1, 1, name="first")
            keep_bodies(first)
            second = place(box(1, 1, 1, name="second"),
                           minAt(atMid(first)), midAt(atMid(first)), midAt(atMid(first)))
            intersection(first, second, name="intersection")

    def test_keep_base_recursive(self):
        with keep_subtree(False):
            first = box(1, 1, 1, name="first")
            keep_bodies(first)
            second = place(box(1, 1, 1, name="second"),
                           minAt(atMid(first)), midAt(atMid(first)), midAt(atMid(first)))
            intersection1 = intersection(first, second, name="intersection1")

            third = place(box(1, 1, 1, name="third"),
                          midAt(atMid(first)), midAt(atMid(first)), maxAt(atMid(first)))
            intersection(intersection1, third, name="intersection2")

    def test_keep_tool(self):
        with keep_subtree(False):
            first = box(1, 1, 1, name="first")
            second = place(box(1, 1, 1, name="second"),
                           minAt(atMid(first)), midAt(atMid(first)), midAt(atMid(first)))
            keep_bodies(second)
            intersection(first, second, name="intersection")

    def test_keep_tool_recursive(self):
        with keep_subtree(False):
            first = box(1, 1, 1, name="first")
            second = place(box(1, 1, 1, name="second"),
                           minAt(atMid(first)), midAt(atMid(first)), midAt(atMid(first)))
            keep_bodies(second)
            intersection1 = intersection(first, second, name="intersection1")

            third = place(box(1, 1, 1, name="third"),
                          midAt(atMid(second)), midAt(atMid(second)), maxAt(atMid(second)))
            intersection(intersection1, third, name="intersection2")

    def test_deep_keep(self):
        set_parametric(True)
        first = box(1, 1, 1, name="first")
        second = place(box(1, 1, 1, name="second"),
                       minAt(atMid(first)), midAt(atMid(first)), midAt(atMid(first)))
        keep_bodies(first)
        intersection1 = intersection(first, second, name="intersection1")
        with keep_subtree(False):
            third = place(box(1, 1, 1, name="third"),
                          midAt(atMid(first)), midAt(atMid(first)), maxAt(atMid(first)))
            intersection2 = intersection(intersection1, third, name="intersection2")

    def test_keep_subtree(self):
        set_parametric(True)
        first = box(1, 1, 1, name="first")
        second = place(box(1, 1, 1, name="second"),
                       minAt(atMid(first)), midAt(atMid(first)), midAt(atMid(first)))
        intersection1 = intersection(first, second, name="intersection1")
        keep_bodies(intersection1)
        with keep_subtree(False):
            third = place(box(1, 1, 1, name="third"),
                          midAt(atMid(first)), midAt(atMid(first)), maxAt(atMid(first)))
            intersection2 = intersection(intersection1, third, name="intersection2")


    def test_keep_duplicated_tool(self):
        with keep_subtree(False):
            first = box(1, 1, 1, name="first")
            second = place(box(1, 1, 1, name="second"),
                           minAt(atMid(first)), midAt(atMid(first)), midAt(atMid(first)))
            keep_bodies(second)
            dup = duplicate(lambda o, v: rz(o, v, center=midOf(first).asArray()), (0, -90), second)

            intersection1 = intersection(first, dup, name="intersection1")

            third = place(box(1, 1, 1, name="third"),
                          maxAt(atMid(second)), minAt(atMin(second)), maxAt(atMid(second)))
            intersection(intersection1, third, name="intersection2")

    def test_keep_duplicated_target(self):
        with keep_subtree(False):
            first = box(1, 1, 1, name="first")
            keep_bodies(first)
            dup = duplicate(tx, (0, 2, 4, 6, 8), first)

            second = place(box(10, .5, .5),
                           minAt(atMin(first)), midAt(atMid(first)), midAt(atMid(first)))

            intersection1 = intersection(dup, second, name="intersection")


from test_utils import load_tests
def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]
                                                                #, pattern="keep_duplicated_target"
                                                                )
    unittest.TextTestRunner(failfast=True).run(test_suite)
