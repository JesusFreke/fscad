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
from adsk.core import Vector3D

import math
import unittest

# note: load_tests is required for the "pattern" test filtering functionality in loadTestsFromModule in run()
from fscad.test_utils import FscadTestCase, load_tests
from fscad.fscad import *


class FilletChamferTest(FscadTestCase):
    def test_basic_fillet(self):
        box = Box(1, 1, 1)
        fillet = Fillet(box.shared_edges(box.front, box.left), .25)
        fillet.create_occurrence(True)

    def test_two_edge_fillet(self):
        box = Box(1, 1, 1)
        fillet = Fillet(box.shared_edges(box.front, [box.left, box.right]), .25)
        fillet.create_occurrence(True)

    def test_two_body_fillet(self):
        rect = Rect(1, 1)
        rect2 = rect.copy()
        rect2.tx(2)
        extrude = Extrude(Union(rect, rect2), 1)
        fillet = Fillet(extrude.shared_edges(extrude.end_faces, extrude.side_faces), .25)
        fillet.create_occurrence(True)

    def test_smooth_fillet(self):
        box = Box(1, 1, 1)
        fillet = Fillet(box.shared_edges([box.front, box.top, box.left], [box.front, box.top, box.left]), .25, True)
        fillet.create_occurrence(True)

    def test_basic_chamfer(self):
        box = Box(1, 1, 1)
        chamfer = Chamfer(box.shared_edges(box.front, box.left), .25)
        chamfer.create_occurrence(True)

    def test_two_edge_chamfer(self):
        box = Box(1, 1, 1)
        chamfer = Chamfer(box.shared_edges(box.front, [box.left, box.right]), .25)
        chamfer.create_occurrence(True)

    def test_two_body_chamfer(self):
        rect = Rect(1, 1)
        rect2 = rect.copy()
        rect2.tx(2)
        extrude = Extrude(Union(rect, rect2), 1)
        chamfer = Chamfer(extrude.shared_edges(extrude.end_faces, extrude.side_faces), .25)
        chamfer.create_occurrence(True)

    def test_uneven_chamfer(self):
        box = Box(1, 1, 1)
        chamfer = Chamfer(box.shared_edges(box.front, box.left), .25, .5)
        chamfer.create_occurrence(True)

    def test_chamfered_faces(self):
        box = Box(1, 1, 1)
        chamfer = Chamfer(box.shared_edges(box.top, [box.left, box.right, box.front, box.back]), .25)
        chamfer.create_occurrence(True)

        self.assertEqual(len(chamfer.chamfered_faces), 4)
        for face in chamfer.chamfered_faces:
            self.assertEqual(
                math.degrees(face.get_plane().normal.angleTo(Vector3D.create(0, 0, 1))),
                45)


def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
