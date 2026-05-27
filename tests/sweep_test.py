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
import math

import adsk.fusion
from adsk.core import Line3D, Point3D, Vector3D

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

    def test_helical_sweep_without_guide(self):
        profile = Rect(1, 1)

        helix_radius = 10
        profile.place(
            -profile == helix_radius,
            ~profile == 0,
            -profile == 0)

        helix_angle = 35

        profile.rx(-helix_angle, center=(0, 0, 0))

        helical_pitch = 2 * math.pi * helix_radius * math.tan(math.radians((90-helix_angle)))

        helix = brep().createHelixWire(
            axisPoint=Point3D.create(0, 0, 0),
            axisVector=Vector3D.create(0, 0, 1),
            startPoint=Point3D.create(helix_radius, 0, 0),
            pitch=helical_pitch,
            turns=2,
            taperAngle=0)

        # without a guide rail, the profile does a weird twist while it's sweeping, so it doesn't stay in the same
        # orientation, relative to the surface of the cylinder the hexix circumscribes
        Sweep(profile, [helix.edges[0].geometry]).create_occurrence(scale=.1)

    def test_helical_sweep_with_guide(self):
        profile = Rect(1, 1)

        helix_radius = 10
        profile.place(
            -profile == helix_radius,
            ~profile == 0,
            -profile == 0)

        helix_angle = 35

        profile.rx(-helix_angle, center=(0, 0, 0))

        helical_pitch = 2 * math.pi * helix_radius * math.tan(math.radians((90-helix_angle)))

        helix = brep().createHelixWire(
            axisPoint=Point3D.create(0, 0, 0),
            axisVector=Vector3D.create(0, 0, 1),
            startPoint=Point3D.create(helix_radius, 0, 0),
            pitch=helical_pitch,
            turns=2,
            taperAngle=0)

        guide_helix = brep().createHelixWire(
            axisPoint=Point3D.create(0, 0, 0),
            axisVector=Vector3D.create(0, 0, 1),
            startPoint=Point3D.create(helix_radius + profile.size().x/2, 0, 0),
            pitch=helical_pitch,
            turns=2,
            taperAngle=0)

        # with a guide rail, the profile stays in the same orientation relative to the surface of the cylinder the
        # helix circumscribes. This is useful for helical gears/threads, etc.
        Sweep(profile, [helix.edges[0].geometry], [guide_helix.edges[0].geometry]).create_occurrence(scale=.1)


def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__],
                                                                #pattern="revolve_component_with_multiple_faces"
                                                                )
    unittest.TextTestRunner(failfast=True).run(test_suite)
