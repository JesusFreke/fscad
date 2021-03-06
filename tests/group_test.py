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

from fscad import *

import adsk.fusion
import unittest
import test_utils
import importlib
importlib.reload(test_utils)
import test_utils


class GroupTest(test_utils.FscadTestCase):
    def validate_test(self):
        if self._test_name == "components":
            return

    def test_group(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        box2.place((-box2 == +box1) + 1)

        bounding_box = box1.bounding_box.raw_bounding_box
        bounding_box.expand(box2.right.brep.pointOnFace)
        total_bounding_box = BoundingBox(bounding_box)

        void = Difference(total_bounding_box.make_box(), box1, box2)

        group = Group([box1, box2], [void])
        group.create_occurrence(True)

    def test_group_no_children(self):
        box1 = Box(1, 1, 1, "box1")
        box2 = Box(1, 1, 1, "box2")
        box2.place((-box2 == +box1) + 1)

        bounding_box = box1.bounding_box.raw_bounding_box
        bounding_box.expand(box2.right.brep.pointOnFace)
        total_bounding_box = BoundingBox(bounding_box)

        void = Difference(total_bounding_box.make_box(), box1, box2)

        group = Group([box1, box2], [void])
        group.create_occurrence(False)

    def test_group_rotation(self):
        box1 = Box(1, 1, 10, "box1")
        box2 = Box(1, 1, 10, "box2")
        box2.place((-box2 == +box1) + 1)

        bounding_box = box1.bounding_box.raw_bounding_box
        bounding_box.expand(box2.right.brep.pointOnFace)
        total_bounding_box = BoundingBox(bounding_box)

        void = Difference(total_bounding_box.make_box(), box1, box2)

        group = Group([box1, box2], [void])
        group.ry(45)
        group.create_occurrence(True)

    def test_planar_group(self):
        rect1 = Rect(1, 1)
        rect2 = Rect(1, 1)

        rect2.place(-rect2 == +rect1,
                    ~rect2 == ~rect1,
                    ~rect2 == ~rect1)

        group = Group([rect1, rect2])
        group.create_occurrence(True)

        self.assertIsNotNone(group.get_plane())
        self.assertTrue(group.get_plane().isCoPlanarTo(rect1.get_plane()))

    def test_non_planar_group(self):
        rect1 = Rect(1, 1)
        rect2 = Rect(1, 1)

        rect2.place(-rect2 == +rect1,
                    ~rect2 == ~rect1,
                    ~rect2 == ~rect1)

        rect2.ry(45, center=rect2.min())

        group = Group([rect1, rect2])
        group.create_occurrence(True)

        self.assertIsNone(group.get_plane())

    def test_components(self):
        rect1 = Rect(1, 1)
        rect2 = Rect(1, 1)
        rect2.tx(2)
        group = Group([rect1, rect2])

        for body in group.bodies:
            self.assertEqual(body.component, group)

        for face in group.faces:
            self.assertEqual(face.component, group)

        for edge in group.edges:
            self.assertEqual(edge.component, group)

    def test_named_faces_on_children(self):
        box1 = Box(1, 1, 1, name="box1")
        box2 = Box(1, 1, 1, name="box2")
        box2.tx(2)

        box1.add_named_faces("top", box1.top)

        group = Group([box1, box2])

        self.assertEqual(group.find_children("box1", recursive=False)[0].named_faces("top")[0].mid().asArray(),
                         (.5, .5, 1.0))

        group.tx(10)

        self.assertEqual(group.find_children("box1", recursive=False)[0].named_faces("top")[0].mid().asArray(),
                         (10.5, .5, 1.0))


from test_utils import load_tests
def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(
        sys.modules[__name__]
        #, pattern="named_faces_on_children"
        )
    unittest.TextTestRunner(failfast=True).run(test_suite)
