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


from test_utils import load_tests
def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(
        sys.modules[__name__]
        #, pattern="bounding_boxes_intersect_but_geometry_doesnt"
        )
    unittest.TextTestRunner(failfast=True).run(test_suite)
