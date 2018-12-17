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


class UnionTest(test_utils.FscadTestCase):
    def test_basic_union(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        box2.place(-box2 == +box1,
                   ~box2 == ~box1,
                   ~box2 == ~box1)
        union = Union(box1, box2)

        self.assertEqual(union.size().asArray(), (2, 1, 1))
        self.assertEqual(union.min().asArray(), (0, 0, 0))
        self.assertEqual(union.mid().asArray(), (1, .5, .5))
        self.assertEqual(union.max().asArray(), (2, 1, 1))

        union.create_occurrence()

    def test_union_children(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        box2.place(-box2 == +box1,
                   ~box2 == ~box1,
                   ~box2 == ~box1)
        union = Union(box1, box2, name="union")

        box3 = Box(1, 1, 1, "box3")
        box3.place(-box3 == +union,
                   ~box3 == ~union,
                   ~box3 == ~union)

        union2 = Union(union, box3, name="union2")

        union2.create_occurrence(True)

    def test_union_add(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        box2.place(-box2 == +box1,
                   ~box2 == ~box1,
                   ~box2 == ~box1)
        union = Union(box1, box2)

        box3 = Box(1, 1, 1, "box3")
        box3.place(-box3 == +union,
                   ~box3 == ~union,
                   ~box3 == ~union)
        union.add(box3)

        union.create_occurrence(True)

    def test_union_subcomponent(self):
        box1 = Box(2, 1, 1, "Box1")
        box2 = Box(1, 2, 1, "Box2")
        box2.place(~box2 == ~box1,
                   -box2 == +box1,
                   ~box2 == ~box1)
        union = Union(box1, box2)
        box3 = Box(1, 2, 1, "Box3")
        box3.place(~box3 == ~box1,
                   +box3 == -box1,
                   ~box3 == ~box1)
        union2 = Union(box1, box3)
        union2.place(z=-union2 == +box1)

        union.create_occurrence(True)
        union2.create_occurrence(True)

from test_utils import load_tests
def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
