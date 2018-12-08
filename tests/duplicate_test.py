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


class DuplicateTest(test_utils.FscadTestCase):
    def test_simple_duplicate(self):
        first = box(1, 1, 1, name="first")
        second = duplicate_of(first)
        translate(second, 5, 5, 5)

        place(union(*find_all_duplicates(first)),
              midAt(0), midAt(0), midAt(0))

    def test_duplicate_holes(self):

        def subcomponent():
            body = cylinder(2, 2, name="body")
            leg = place(cylinder(10, .25, name="leg"),
                        midAt(atMid(body)), midAt(atMid(body)), maxAt(atMax(body)))
            legs = duplicate(tx, (-1, 1), leg)
            diff = difference(body, legs)
            return diff, legs

        base = box(20, 20, 2, name="base")
        sub_positive, sub_negative = subcomponent()
        sub = place(sub_positive,
                    maxAt(lambda i: atMax(base)(i) - .5), midAt(atMid(base)), minAt(atMax(base)))
        subs = duplicate(lambda o, v: rz(o, v, center=midOf(base).asArray()), (0, 90, 180, 270), sub_positive)

        all_negatives = find_all_duplicates(sub_negative)
        with Joiner(union, name="leg_holes") as join:
            for dup in all_negatives:
                join(duplicate_of(dup))
        difference(base, join.result())


from test_utils import load_tests
def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
