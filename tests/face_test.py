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

import unittest

# note: load_tests is required for the "pattern" test filtering functionality in loadTestsFromModule in run()
from fscad.test_utils import FscadTestCase, load_tests
from fscad.fscad import *


class FaceTest(FscadTestCase):
    def validate_test(self):
        if self._test_name in ("make_component", "loops"):
            return super().validate_test()
        return

    def test_adjacent_coincident_faces(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        box2.place(-box2 == +box1,
                   ~box2 == ~box1,
                   ~box2 == ~box1)

        faces = box1.find_faces(box2.left)
        self.assertEqual(len(faces), 1)
        self.assertTrue(faces[0].brep.geometry.normal.isParallelTo(Vector3D.create(1, 0, 0)))
        self.assertEqual(faces[0].brep.pointOnFace.x, 1)

    def test_diff_coincident_face(self):
        box = Box(1, 1, 1)
        hole = Box(.5, .5, .5, "hole")
        hole.place(+hole == +box,
                   ~hole == ~box,
                   ~hole == ~box)
        diff = Difference(box, hole)

        faces = diff.find_faces(hole.left)
        self.assertEqual(len(faces), 1)
        self.assertTrue(faces[0].brep.geometry.normal.isParallelTo(Vector3D.create(1, 0, 0)))
        self.assertEqual(faces[0].brep.pointOnFace.x, .5)

        faces = diff.find_faces(hole)
        self.assertEqual(len(faces), 5)
        for face in faces:
            self.assertEqual(face.brep.area, .5*.5)

        faces = diff.find_faces([hole])
        self.assertEqual(len(faces), 5)
        for face in faces:
            self.assertEqual(face.brep.area, .5*.5)

        faces = diff.find_faces([list(hole.bodies[0].faces)])
        self.assertEqual(len(faces), 5)
        for face in faces:
            self.assertEqual(face.brep.area, .5*.5)

        faces = diff.find_faces(hole.bodies[0])
        self.assertEqual(len(faces), 5)
        for face in faces:
            self.assertEqual(face.brep.area, .5*.5)

    def test_spherical_coincident_face(self):
        box = Box(1, 1, 1)
        sphere = Sphere(.25)
        sphere.place(~sphere == +box,
                     ~sphere == ~box,
                     ~sphere == ~box)
        diff = Difference(box, sphere)

        faces = diff.find_faces(sphere.surface)
        self.assertEqual(len(faces), 1)
        self.assertTrue(isinstance(faces[0].brep.geometry, adsk.core.Sphere))

    def test_face_with_multiple_coincident_faces(self):
        box = Box(1, 1, 1)
        selector = Box(1, 1, 1)
        selector_cut = Box(.25, .25, 1)
        selector_cut.place(
            ~selector_cut == ~selector,
            -selector_cut == -selector,
            -selector_cut == -selector)

        selector = Difference(selector, selector_cut)

        selector.place(
            ~selector == ~box,
            -selector == +box,
            -selector == -box)

        faces = box.find_faces(selector)
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0].brep, box.back.brep)

    def test_connected_faces(self):
        box = Box(1, 1, 1)

        self.assertSetEqual({face.brep.tempId for face in box.bottom.connected_faces},
                            {face.brep.tempId for face in [box.left, box.right, box.front, box.back]})
        self.assertSetEqual({face.brep.tempId for face in box.top.connected_faces},
                            {face.brep.tempId for face in [box.left, box.right, box.front, box.back]})
        self.assertSetEqual({face.brep.tempId for face in box.left.connected_faces},
                            {face.brep.tempId for face in [box.top, box.bottom, box.front, box.back]})
        self.assertSetEqual({face.brep.tempId for face in box.right.connected_faces},
                            {face.brep.tempId for face in [box.top, box.bottom, box.front, box.back]})
        self.assertSetEqual({face.brep.tempId for face in box.front.connected_faces},
                            {face.brep.tempId for face in [box.top, box.bottom, box.left, box.right]})
        self.assertSetEqual({face.brep.tempId for face in box.back.connected_faces},
                            {face.brep.tempId for face in [box.top, box.bottom, box.left, box.right]})

    def test_component_faces(self):
        box1 = Box(1, 1, 1, name="box1")
        box2 = Box(1, 1, 1, name="box2")
        box2.place(
            (-box2 == +box1) + 5,
            ~box2 == ~box1,
            ~box2 == ~box1)

        self.assertEquals(len(Group([box1, box2]).faces), 12)

    def test_face_after_translate(self):
        box = Box(1, 1, 1)
        face = box.top

        self.assertEqual(face.mid().asArray(), (.5, .5, 1))

        box.tx(1)

        self.assertEqual(face.mid().asArray(), (1.5, .5, 1))

    def test_make_component(self):
        cylinder = Cylinder(1, 1)
        cylinder.side.make_component(name="CylinderFace").create_occurrence(create_children=True)

    def test_loops(self):
        box = Box(1, 1, 1)
        inner_box = Box(.5, .5, 1)

        inner_box.place(
            ~inner_box == ~box,
            ~inner_box == ~box,
            ~inner_box == ~box)

        difference = Difference(box, inner_box)
        top_face = difference.find_faces(box.top)[0]

        self.assertEqual(len(top_face.loops), 2)

        outer_loop = top_face.loops[0] if top_face.loops[0].is_outer else top_face.loops[1]
        inner_loop = top_face.loops[1] if top_face.loops[0].is_outer else top_face.loops[0]

        self.assertEqual(outer_loop.size().asArray(), (1, 1, 0))
        self.assertEqual(inner_loop.size().asArray(), (.5, .5, 0))

        self.assertEqual(len(outer_loop.edges), 4)
        self.assertEqual(len(inner_loop.edges), 4)

        Silhouette(outer_loop, box.top.get_plane()).create_occurrence(True)
        Silhouette(inner_loop, box.top.get_plane()).create_occurrence(True)

    def test_reversed_normal(self):
        rect = Rect(1, 1)
        rect.create_occurrence(scale=.1)

        extrude = Extrude(rect, 1)
        extrude.create_occurrence(scale=.1)

        self.assertTrue(extrude.start_faces[0].brep.isParamReversed)
        self.assertGreater(extrude.start_faces[0].brep.geometry.normal.z, 0)
        self.assertLess(extrude.start_faces[0].get_plane().normal.z, 0)


def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__],
                                                                #pattern="connected_faces",
                                                                )
    unittest.TextTestRunner(failfast=True).run(test_suite)
