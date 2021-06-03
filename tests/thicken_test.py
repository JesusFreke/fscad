# Copyright 2020 Google LLC
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

# note: load_tests is required for the "pattern" test filtering functionality in loadTestsFromModule in run()
from fscad.test_utils import FscadTestCase, load_tests
from fscad.fscad import *


class ThickenTest(FscadTestCase):
    def test_basic_cylinder(self):
        cylinder = Cylinder(1, 1)
        Thicken(cylinder.side, 1).create_occurrence(create_children=True)

    def test_quarter_cylinder(self):
        cylinder = Cylinder(1, 1)
        box = Box(1, 1, 1)
        box.place(
            -box == ~cylinder,
            -box == ~cylinder,
            -box == -cylinder)
        assembly = Intersection(cylinder, box)
        Thicken(assembly.find_faces(cylinder.side)[0], 1).create_occurrence(create_children=True)

    def test_cylinder_with_hole(self):
        cylinder = Cylinder(1, 1)

        hole = Cylinder(1, .25, name="hole")
        hole.rx(90)
        hole.place(
            ~hole == ~cylinder,
            +hole == ~cylinder,
            ~hole == ~cylinder)

        assembly = Difference(cylinder, hole)

        Thicken(assembly.find_faces(cylinder.side)[0], 1).create_occurrence(create_children=True)

    def test_rotated_quarter_cylinder(self):
        cylinder = Cylinder(1, 1)
        box = Box(1, 1, 1)
        box.place(
            -box == ~cylinder,
            -box == ~cylinder,
            -box == -cylinder)
        assembly = Intersection(cylinder, box)
        assembly.ry(45)

        Thicken(assembly.find_faces(cylinder.side)[0], 1).create_occurrence(create_children=True)

    def test_translated_quarter_cylinder(self):
        cylinder = Cylinder(1, 1)
        box = Box(1, 1, 1)
        box.place(
            -box == ~cylinder,
            -box == ~cylinder,
            -box == -cylinder)
        assembly = Intersection(cylinder, box)
        assembly.tx(.5)

        Thicken(assembly.find_faces(cylinder.side)[0], 1).create_occurrence(create_children=True)

    def test_truncated_cone(self):
        cone = Cylinder(1, 1, .5, name="cone")
        Thicken(cone.side, 1).create_occurrence(create_children=True)

    def test_full_cone(self):
        cone = Cylinder(1, 1, 0, name="cone")
        Thicken(cone.side, 1).create_occurrence(create_children=True)

    def test_cylindrical_face(self):
        cylinder = Cylinder(1, 1)
        Thicken(cylinder.side.make_component(), 1).create_occurrence(create_children=True)

    def test_box_negative_thickness(self):
        box = Box(1, 1, 1)
        Thicken(box.top, -1.5).create_occurrence(create_children=True)

    def test_cylinder_face_large_negative_thickness(self):
        cylinder = Cylinder(1, 1)
        Thicken(cylinder.side, -3).create_occurrence(create_children=True)

    def test_cylinder_face_small_negative_thickness(self):
        cylinder = Cylinder(1, 1)
        Thicken(cylinder.side, -.5).create_occurrence(create_children=True)

    def test_thicken_body(self):
        box = Box(1, 1, 1)
        Thicken(box, 1).create_occurrence(create_children=True)

    def test_multiple_components(self):
        box1 = Box(1, 1, 1)
        box2 = Box(1, 1, 1)
        box2.place(
            (-box2 == +box1) + .1,
            ~box2 == ~box1,
            ~box2 == ~box1)
        Thicken((box1, box2), 1).create_occurrence(create_children=True)

    def test_multiple_faces(self):
        box = Box(1, 1, 1)
        Thicken((box.front, box.top), 1).create_occurrence(create_children=True)


def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(
        sys.modules[__name__]
        #, pattern="multiple_faces"
    )

    unittest.TextTestRunner(failfast=True).run(test_suite)
