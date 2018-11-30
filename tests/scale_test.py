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


class ScaleTest(test_utils.FscadTestCase):
    def test_simple_scale(self):
        first = box(1, 1, 1, name="first")
        scale(first, 2)

    def test_non_uniform_scale(self):
        first = sphere(1, name="first")
        scale(first, (1, 1, .1))

    def test_duplicates_scale(self):
        first = box(1, 1, 1, name="first")
        dup = duplicate(tx, (0, 2, 4, 6, 8), first)
        scale(dup, 2)

    def test_duplicate_non_uniform_scale(self):
        first = box(1, 1, 1, name="first")
        dup = duplicate(tx, (0, 2, 4, 6, 8), first)
        try:
            scale(dup, (2, 3, 4))
            self.fail("Expected error did not occur")
        except ValueError:
            pass

    def test_scale_with_center(self):
        first = box(1, 1, 1, name="first")
        scale(first, 2, (1, 1, 1))

    def test_non_uniform_scale_with_center(self):
        first = box(1, 1, 1, name="first")
        scale(first, (1, 1, .1), (1, 1, 1))

    def test_simple_mirror(self):
        first = box(1, 1, 1, name="first")
        scale(first, (-1, 1, 1))

    def test_mirror_with_center(self):
        first = box(1, 1, 1, name="first")
        scale(first, (-1, 1, 1), center=(1, 1, 1))

    def test_duplicates_mirror(self):
        first = box(1, 1, 1, name="first")
        dup = duplicate(tx, (0, 2, 4, 6, 8), first)
        scale(dup, (-1, 1, 1))

    def test_non_uniform_mirror(self):
        first = box(1, 1, 1, name="first")
        try:
            scale(first, (1, 1, -10), (1, 1, 1))
            self.fail("Expected error did not occur")
        except ValueError:
            pass


def run(context):
    test_suite = unittest.defaultTestLoader.loadTestsFromTestCase(ScaleTest)
    unittest.TextTestRunner(failfast=True).run(test_suite)
