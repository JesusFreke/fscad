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


class FilletChamferTest(test_utils.FscadTestCase):
    def test_simple_fillet(self):
        first = box(1, 1, 1, name="first")

        result = edges(first, ["top"], ["right"])

        fillet(result, .1)

    def test_multiple_edge_fillet(self):
        first = box(1, 1, 1, name="first")

        faces = ["top", "right", "front"]
        result = edges(first, faces, faces)

        fillet(result, .1)

    def test_blended_multiple_edge_fillet(self):
        first = box(1, 1, 1, name="first")

        faces = ["top", "right", "front"]
        result = edges(first, faces, faces)

        fillet(result, .1, True)

    def test_simple_chamfer(self):
        first = box(1, 1, 1, name="first")

        result = edges(first, ["top"], ["right"])

        chamfer(result, .1)

    def test_multiple_edge_chamfer(self):
        first = box(1, 1, 1, name="first")

        faces = ["top", "right", "front"]
        result = edges(first, faces, faces)

        chamfer(result, .1)

    def test_two_distance_chamfer(self):
        first = box(1, 1, 1, name="first")

        faces = ["top", "right", "front"]
        result = edges(first, faces, faces)

        chamfer(result, .2, .1)

    def test_hidden_occurrence_fillet(self):
        first = box(1, 1, 1, name="first")
        second = place(box(.5, .5, .5, name="second"),
                       maxAt(atMax(first)), midAt(atMid(first)), midAt(atMid(first)))
        diff = difference(first, second, name="diff")

        result = edges(first, ["top"], ["right"])
        got_error = False
        try:
            fillet(result, .1)
        except:
            got_error = True
        self.assertTrue(got_error)

    def test_hidden_occurrence_chamfer(self):
        first = box(1, 1, 1, name="first")
        second = place(box(.5, .5, .5, name="second"),
                       maxAt(atMax(first)), midAt(atMid(first)), midAt(atMid(first)))
        diff = difference(first, second, name="diff")

        result = edges(first, ["top"], ["right"])
        got_error = False
        try:
            chamfer(result, .1)
        except:
            got_error = True
        self.assertTrue(got_error)


def run(context):
    test_suite = unittest.defaultTestLoader.loadTestsFromTestCase(FilletChamferTest)
    unittest.TextTestRunner(failfast=True).run(test_suite)
