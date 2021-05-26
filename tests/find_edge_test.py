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

from fscad.test_utils import FscadTestCase, unordered_compare
from fscad.fscad import *


class FindEdgeTest(FscadTestCase):
    def validate_test(self):
        pass

    def test_find_edges_2d_with_body_face_intersection(self):
        rect = Rect(1, 1)
        finder = Box(.1, .1, .1)

        finder.place(
            ~finder == ~rect,
            ~finder == +rect,
            -finder == ~rect)

        edges = rect.find_edges(finder)

        self.assertEqual(len(edges), 1)
        self.assertTrue(isinstance(edges[0].brep.geometry, adsk.core.Line3D))
        self.assertEqual(edges[0].brep.geometry.startPoint.asArray(), (1, 1, 0))
        self.assertEqual(edges[0].brep.geometry.endPoint.asArray(), (0, 1, 0))

    def test_find_edges_2d_with_body_edge_intersection(self):
        rect = Rect(1, 1)
        finder = Box(.1, .1, .1)

        finder.place(
            ~finder == ~rect,
            -finder == +rect,
            -finder == ~rect)

        edges = rect.find_edges(finder)

        self.assertEqual(len(edges), 1)
        self.assertTrue(isinstance(edges[0].brep.geometry, adsk.core.Line3D))
        self.assertEqual(edges[0].brep.geometry.startPoint.asArray(), (1, 1, 0))
        self.assertEqual(edges[0].brep.geometry.endPoint.asArray(), (0, 1, 0))

    def test_find_edges_2d_with_face(self):
        rect = Rect(1, 1)
        finder = Box(.1, .1, .1)

        finder.place(
            ~finder == ~rect,
            ~finder == +rect,
            -finder == ~rect)

        edges = rect.find_edges(finder.bottom)

        self.assertEqual(len(edges), 1)
        self.assertTrue(isinstance(edges[0].brep.geometry, adsk.core.Line3D))
        self.assertEqual(edges[0].brep.geometry.startPoint.asArray(), (1, 1, 0))
        self.assertEqual(edges[0].brep.geometry.endPoint.asArray(), (0, 1, 0))

    def test_find_edges_2d_with_edge(self):
        rect = Rect(1, 1)
        finder = Box(.1, .1, .1)

        finder.place(
            ~finder == ~rect,
            +finder == +rect,
            -finder == ~rect)

        edges = rect.find_edges(finder.shared_edges(finder.bottom, finder.back))

        self.assertEqual(len(edges), 1)
        self.assertTrue(isinstance(edges[0].brep.geometry, adsk.core.Line3D))
        self.assertEqual(edges[0].brep.geometry.startPoint.asArray(), (1, 1, 0))
        self.assertEqual(edges[0].brep.geometry.endPoint.asArray(), (0, 1, 0))

    def test_find_edges_2d_point_intersection_not_included(self):
        rect = Rect(1, 1)
        finder = Box(.1, .1, .1)

        finder.place(
            ~finder == ~rect,
            ~finder == +rect,
            -finder == ~rect)

        edges = rect.find_edges(finder.left)

        self.assertEqual(len(edges), 0)

    def test_find_edges_3d(self):
        box = Box(1, 1, 1)
        finder = Box(.1, .1, .1)

        finder.place(
            ~finder == ~box,
            ~finder == +box,
            -finder == +box)

        edges = box.find_edges(finder)

        self.assertEqual(len(edges), 1)
        self.assertTrue(isinstance(edges[0].brep.geometry, adsk.core.Line3D))
        self.assertEqual(edges[0].brep.geometry.startPoint.asArray(), (1, 1, 1))
        self.assertEqual(edges[0].brep.geometry.endPoint.asArray(), (0, 1, 1))

    def test_find_edges_multiple(self):
        box = Box(1, 1, 1)
        finder = Box(.1, .1, .1)

        finder.place(
            ~finder == +box,
            ~finder == +box,
            -finder == +box)

        edges = box.find_edges(finder)

        self.assertEqual(len(edges), 2)
        unordered_compare([edge.brep for edge in edges],
                          box.shared_edges(box.top, [box.back, box.right]))


def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
