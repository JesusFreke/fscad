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

        face1 = get_face(first, "top")
        face2 = get_face(first, "right")

        edges = get_edges([face1], [face2])

        fillet(edges, .1)

    def test_multiple_edge_fillet(self):
        first = box(1, 1, 1, name="first")

        face1 = get_face(first, "top")
        face2 = get_face(first, "right")
        face3 = get_face(first, "front")

        edges = get_edges([face1, face2, face3], [face1, face2, face3])

        fillet(edges, .1)

    def test_blended_multiple_edge_fillet(self):
        first = box(1, 1, 1, name="first")

        face1 = get_face(first, "top")
        face2 = get_face(first, "right")
        face3 = get_face(first, "front")

        edges = get_edges([face1, face2, face3], [face1, face2, face3])

        fillet(edges, .1, True)

    def test_simple_chamfer(self):
        first = box(1, 1, 1, name="first")

        face1 = get_face(first, "top")
        face2 = get_face(first, "right")

        edges = get_edges([face1], [face2])

        chamfer(edges, .1)

    def test_multiple_edge_chamfer(self):
        first = box(1, 1, 1, name="first")

        face1 = get_face(first, "top")
        face2 = get_face(first, "right")
        face3 = get_face(first, "front")

        edges = get_edges([face1, face2, face3], [face1, face2, face3])

        chamfer(edges, .1)

    def test_two_distance_chamfer(self):
        first = box(1, 1, 1, name="first")

        face1 = get_face(first, "top")
        face2 = get_face(first, "right")
        face3 = get_face(first, "front")

        edges = get_edges([face1, face2, face3], [face1, face2, face3])

        chamfer(edges, .2, .1)


def run(context):
    test_suite = unittest.defaultTestLoader.loadTestsFromTestCase(FilletChamferTest)
    unittest.TextTestRunner(failfast=True).run(test_suite)
