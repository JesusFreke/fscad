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


class TranslateTest(test_utils.FscadTestCase):
    def test_translate_x(self):
        translate(box(1, 2, 3), x=5)

    def test_translate_y(self):
        translate(box(1, 2, 3), y=5)

    def test_translate_z(self):
        translate(box(1, 2, 3), z=5)

    def test_translate_all(self):
        translate(box(1, 2, 3), x=1, y=2, z=3)

    def test_translate_all_sequentially(self):
        r1 = translate(box(1, 2, 3), z=3)
        r2 = translate(r1, y=2)
        r3 = translate(r2, x=1)

    def test_tx(self):
        tx(box(1, 2, 3), 5)

    def test_ty(self):
        ty(box(1, 2, 3), 5)

    def test_tz(self):
        tz(box(1, 2, 3), 5)

    def test_txtytz(self):
        tz(ty(tx(box(1, 2, 3), 1), 2), 3)

    def test_translate_rotated(self):
        tx(rz(box(1, 2, 3), 45), 10)

    def test_hidden_occurrence_translate(self):
        first = box(1, 1, 1, name="first")
        second = place(box(.5, .5, .5, name="second"),
                       maxAt(atMax(first)), midAt(atMid(first)), midAt(atMid(first)))
        diff = difference(first, second)

        got_error = False
        try:
            translate(second, 5)
        except:
            got_error = True
        self.assertTrue(got_error)


def run(context):
    test_suite = unittest.defaultTestLoader.loadTestsFromTestCase(TranslateTest)
    unittest.TextTestRunner(failfast=True).run(test_suite)
