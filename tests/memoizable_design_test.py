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

# note: load_tests is required for the "pattern" test filtering functionality in loadTestsFromModule in run()
from fscad.test_utils import FscadTestCase, load_tests
from fscad.fscad import *


class MemoizableDesignTest(FscadTestCase):
    def test_called_twice(self):

        global times_called
        times_called = 0

        class TestClass(MemoizableDesign):

            @MemoizableDesign.MemoizeComponent
            def test_design(self):
                global times_called
                times_called += 1
                return Box(1, 1, 1)

        test = TestClass()
        box1 = test.test_design()
        box2 = test.test_design()
        box2.tx(5)
        box1.create_occurrence()
        box2.create_occurrence()
        self.assertEquals(times_called, 1)

    def test_called_twice_with_name_as_keyword(self):

        global times_called
        times_called = 0

        class TestClass(MemoizableDesign):

            @MemoizableDesign.MemoizeComponent
            def test_design(self, name):
                global times_called
                times_called += 1
                return Box(1, 1, 1, name=name)

        test = TestClass()
        box1 = test.test_design(name="box1")
        box2 = test.test_design(name="box2")
        box2.tx(5)
        box1.create_occurrence()
        box2.create_occurrence()
        self.assertEquals(times_called, 1)

    def test_called_twice_with_name_as_positional(self):

        global times_called
        times_called = 0

        class TestClass(MemoizableDesign):

            @MemoizableDesign.MemoizeComponent
            def test_design(self, name):
                global times_called
                times_called += 1
                return Box(1, 1, 1, name=name)

        test = TestClass()
        box1 = test.test_design("box1")
        box2 = test.test_design("box2")
        box2.tx(5)
        box1.create_occurrence()
        box2.create_occurrence()
        self.assertEquals(times_called, 1)

    def test_with_arguments(self):

        global times_called
        times_called = 0

        class TestClass(MemoizableDesign):
            @MemoizableDesign.MemoizeComponent
            def test_design(self, tx, name):
                global times_called
                times_called += 1
                box = Box(1, 1, 1, name=name)
                box.tx(tx)
                return box

        test = TestClass()
        box1 = test.test_design(0, "box1")
        box2 = test.test_design(0, "box2")
        box2.tx(5)
        box3 = test.test_design(10, "box3")
        box1.create_occurrence()
        box2.create_occurrence()
        box3.create_occurrence()
        self.assertEquals(times_called, 2)

    def test_different_instances(self):

        global times_called
        times_called = 0

        class TestClass(MemoizableDesign):
            @MemoizableDesign.MemoizeComponent
            def test_design(self, tx, name):
                global times_called
                times_called += 1
                box = Box(1, 1, 1, name=name)
                box.tx(tx)
                return box

        test1 = TestClass()
        box1 = test1.test_design(0, "box1")
        box2 = test1.test_design(0, "box2")
        box2.tx(5)

        test2 = TestClass()
        box3 = test2.test_design(0, "box3")
        box3.tx(10)
        box4 = test2.test_design(0, "box4")
        box4.tx(15)

        box1.create_occurrence()
        box2.create_occurrence()
        box3.create_occurrence()
        box4.create_occurrence()
        self.assertEquals(times_called, 2)


def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__],
                                                                # pattern="called_twice_with_name_as_keyword"
                                                                )
    unittest.TextTestRunner(failfast=True).run(test_suite)
