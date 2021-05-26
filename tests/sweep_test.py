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
from adsk.core import Line3D, Point3D

import unittest

# note: load_tests is required for the "pattern" test filtering functionality in loadTestsFromModule in run()
from fscad.test_utils import FscadTestCase, load_tests
from fscad.fscad import *


class SweepTest(FscadTestCase):
    def test_line_sweep(self):
        rect = Rect(1, 1)

        Sweep(
            rect,
            [Line3D.create(
                rect.mid(),
                Point3D.create(rect.mid().x, rect.mid().y, 1))]).create_occurrence(create_children=True)

    def test_straight_edge_sweep(self):
        rect = Rect(1, 1)
        box = Box(1, 1, 1)
        box.place(
            +box == ~rect,
            +box == ~rect,
            -box == -rect)

        Sweep(
            rect,
            [box.shared_edges(box.front, box.right)[0]]).create_occurrence(create_children=True)

    def test_curved_edge_sweep(self):
        rect = Rect(1, 1)
        circle = Circle(5)
        circle.rx(90)
        circle.place(
            -circle == ~rect,
            ~circle == ~rect,
            ~circle == ~rect)

        Sweep(
            rect,
            [circle.edges[0]]).create_occurrence(create_children=True)

    def test_perpindicular_edge_path_sweep(self):
        rect = Rect(1, 1)
        box = Box(1, 1, 1)
        box.place(
            +box == ~rect,
            +box == ~rect,
            -box == -rect)

        Sweep(
            rect,
            [
                box.shared_edges(box.front, box.right)[0],
                box.shared_edges(box.front, box.top)[0]]).create_occurrence(create_children=True)

    def test_full_twist(self):
        rect = Rect(1, 1)

        Sweep(
            rect,
            [Line3D.create(
                rect.mid(),
                Point3D.create(rect.mid().x, rect.mid().y, 1))],
            turns=1).create_occurrence(create_children=True)

    def test_negative_twist(self):
        rect = Rect(1, 1)

        Sweep(
            rect,
            [Line3D.create(
                rect.mid(),
                Point3D.create(rect.mid().x, rect.mid().y, 1))],
            turns=-1).create_occurrence(create_children=True)

    def test_offset_twist(self):
        circle = Circle(1)

        Sweep(
            circle,
            [Line3D.create(
                Point3D.create(circle.mid().x - .4, circle.mid().y, circle.mid().z),
                Point3D.create(circle.mid().x - .4, circle.mid().y, 1))],
            turns=.5).create_occurrence(create_children=True)


    def test_body_face(self):
        box = Box(1, 1, 1)

        Sweep(
            box.top,
            [
                Line3D.create(
                    box.top.mid(),
                    Point3D.create(
                        box.top.mid().x + 1,
                        box.top.mid().y + 1,
                        box.top.mid().z + 1))
            ],
        ).create_occurrence(create_children=True)

    def test_multiple_faces(self):
        circle = Circle(.5)
        rect = Rect(1, 1)
        rect.place(
            (-rect == +circle) + 1,
            ~rect == ~circle,
            ~rect == ~circle)

        both = Union(circle, rect)

        Sweep(
            both.faces,
            [
                Line3D.create(
                    both.mid(),
                    Point3D.create(both.mid().x, both.mid().y, both.mid().z + 10))
            ],
            turns=1).create_occurrence(create_children=True)


def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__],
                                                                #pattern="revolve_component_with_multiple_faces"
                                                                )
    unittest.TextTestRunner(failfast=True).run(test_suite)
