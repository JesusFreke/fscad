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

import adsk.core
import adsk.fusion

import unittest

# note: load_tests is required for the "pattern" test filtering functionality in loadTestsFromModule in run()
from fscad.test_utils import FscadTestCase, load_tests
from fscad.fscad import *


class OffsetEdgesTest(FscadTestCase):
    def test_positive_offset(self):
        box = Box(1, 1, 1)
        OffsetEdges(box.top, box.top.outer_edges, .5).create_occurrence(True)

    def test_negative_offset(self):
        box = Box(1, 1, 1)
        OffsetEdges(box.top, box.top.outer_edges, -.25).create_occurrence(True)

    def test_partial_profile_negative_offset(self):
        box = Box(1, 1, 1)
        OffsetEdges(box.top,
                    box.shared_edges(box.top, [box.left, box.right, box.front]),
                    -.25).create_occurrence(True)

    def test_partial_profile_negative_offset_forming_closed_loop(self):
        box1 = Box(1, 1, 1)
        box2 = Box(.1, .1, 1)
        box2.place(
            +box2 == +box1,
            -box2 == -box1,
            -box2 == -box1)

        assembly = Difference(box1, box2)

        top_face = assembly.find_faces(box1.top)[0]
        OffsetEdges(top_face,
                assembly.shared_edges(
                    top_face,
                    [
                        *assembly.find_faces(box1.left),
                        *assembly.find_faces(box1.right),
                        *assembly.find_faces(box1.back),
                        *assembly.find_faces(box1.front)
                    ]),
                    -.25).create_occurrence(True)

    def test_partial_profile_positive_offset(self):
        box = Box(1, 1, 1)
        OffsetEdges(box.top,
                    box.shared_edges(box.top, [box.left, box.right, box.front]),
                    .5).create_occurrence(True)

    def test_inner_profile_positive_offset(self):
        box1 = Box(1, 1, 1)
        box2 = Box(.25, .25, 1)
        box2.place(
            ~box2 == ~box1,
            ~box2 == ~box1,
            -box2 == -box1)

        assembly = Difference(box1, box2)

        assembly_top = assembly.find_faces(box1.top)[0]

        OffsetEdges(assembly_top,
                    assembly.shared_edges(
                        assembly_top,
                        assembly.find_faces(box2)),
                    .1).create_occurrence(True)

    def test_inner_profile_negative_offset(self):
        box1 = Box(1, 1, 1)
        box2 = Box(.25, .25, 1)
        box2.place(
            ~box2 == ~box1,
            ~box2 == ~box1,
            -box2 == -box1)

        assembly = Difference(box1, box2)

        assembly_top = assembly.find_faces(box1.top)[0]

        OffsetEdges(assembly_top,
                    assembly.shared_edges(
                        assembly_top,
                        assembly.find_faces(box2)),
                    -.1).create_occurrence(True)

    def test_inner_profile_breaks_out(self):
        box1 = Box(1, 1, 1)
        box2 = Box(.25, .25, 1)
        box2.place(
            ~box2 == ~box1,
            (-box2 == -box1) + .05,
            -box2 == -box1)

        assembly = Difference(box1, box2)

        assembly_top = assembly.find_faces(box1.top)[0]

        OffsetEdges(assembly_top,
                    assembly.shared_edges(
                        assembly_top,
                        assembly.find_faces(box2)),
                    -.1).create_occurrence(True)

    def test_outer_profile_negative_offset_intersects_inner_profile(self):
        box1 = Box(1, 1, 1)
        box2 = Box(.25, .25, 1)
        box2.place(
            ~box2 == ~box1,
            (-box2 == -box1) + .05,
            -box2 == -box1)

        assembly = Difference(box1, box2)

        assembly_top = assembly.find_faces(box1.top)[0]

        OffsetEdges(assembly_top,
                    assembly.shared_edges(
                        assembly_top,
                        assembly.find_faces(box1)),
                    -.1).create_occurrence(True)

    def test_circular_outer_profile(self):
        circle = Circle(1)
        OffsetEdges(circle.faces[0], circle.edges, -.1).create_occurrence(True)

    def test_circular_inner_profile(self):
        outer_circle = Circle(1)
        inner_circle = Circle(.5)
        inner_circle.place(
            ~inner_circle == ~outer_circle,
            ~inner_circle == ~outer_circle,
            ~inner_circle == ~outer_circle)

        assembly = Difference(outer_circle, inner_circle)

        edge_to_offset = None
        for edge in assembly.edges:
            if edge.brep.geometry.radius == .5:
                edge_to_offset = edge

        OffsetEdges(assembly.faces[0], [edge_to_offset], -.1).create_occurrence(True)

    def test_angled_lines(self):
        builder = Builder2D((0, 0))

        builder.line_to((0, 10))
        builder.line_to((2.5, 12))
        builder.line_to((5, 12))
        builder.line_to((7.5, 10))
        builder.line_to((7.5, 0))
        builder.line_to((0, 0))

        shape = builder.build()

        OffsetEdges(
            shape.faces[0],
            [edge for edge in shape.faces[0].edges if not (
                isinstance(edge.brep.geometry, adsk.core.Line3D) and
                edge.brep.geometry.asInfiniteLine().direction.y < app().pointTolerance) and
                edge.mid().x < shape.mid().x],
            -.5).create_occurrence(scale=.1)

def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__],
                                                                #pattern="angled_lines",
                                                                )
    unittest.TextTestRunner(failfast=True).run(test_suite)
