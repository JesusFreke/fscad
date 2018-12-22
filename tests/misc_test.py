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
import fscad

import adsk.core
import adsk.fusion

import unittest
import test_utils
import importlib
importlib.reload(test_utils)
import test_utils
import math


class MiscTest(test_utils.FscadTestCase):
    def validate_test(self):
        pass

    def _do_project_point_to_line_test(self, point: adsk.core.Point3D, line: adsk.core.InfiniteLine3D):
        projection = fscad._project_point_to_line(point, line)
        self.assertTrue(projection.isEqualTo(point) or line.direction.isPerpendicularTo(projection.vectorTo(point)))
        self.assertTrue(projection.isEqualTo(line.origin) or
                        projection.vectorTo(line.origin).isParallelTo(line.direction))

    def test_project_point_to_line_vertical(self):
        self._do_project_point_to_line_test(
            adsk.core.Point3D.create(1, 0, 0),
            adsk.core.InfiniteLine3D.create(
                adsk.core.Point3D.create(0, 0, 0),
                adsk.core.Vector3D.create(0, 0, 1)))

        self._do_project_point_to_line_test(
            adsk.core.Point3D.create(1, 0, 0),
            adsk.core.InfiniteLine3D.create(
                adsk.core.Point3D.create(0, 0, 100),
                adsk.core.Vector3D.create(0, 0, 1)))

        self._do_project_point_to_line_test(
            adsk.core.Point3D.create(1, 0, 0),
            adsk.core.InfiniteLine3D.create(
                adsk.core.Point3D.create(0, 0, 100),
                adsk.core.Vector3D.create(0, 0, -1)))

        self._do_project_point_to_line_test(
            adsk.core.Point3D.create(1, 0, 44732),
            adsk.core.InfiniteLine3D.create(
                adsk.core.Point3D.create(0, 0, 100),
                adsk.core.Vector3D.create(0, 0, 1)))

        self._do_project_point_to_line_test(
            adsk.core.Point3D.create(0, 0, 0),
            adsk.core.InfiniteLine3D.create(
                adsk.core.Point3D.create(0, 0, 100),
                adsk.core.Vector3D.create(0, 0, 1)))

    def test_project_point_to_line_horizontal(self):
        self._do_project_point_to_line_test(
            adsk.core.Point3D.create(0, 0, 1),
            adsk.core.InfiniteLine3D.create(
                adsk.core.Point3D.create(0, 0, 0),
                adsk.core.Vector3D.create(1, 0, 0)))

        self._do_project_point_to_line_test(
            adsk.core.Point3D.create(0, 0, 1),
            adsk.core.InfiniteLine3D.create(
                adsk.core.Point3D.create(100, 0, 0),
                adsk.core.Vector3D.create(1, 0, 0)))

        self._do_project_point_to_line_test(
            adsk.core.Point3D.create(0, 0, 1),
            adsk.core.InfiniteLine3D.create(
                adsk.core.Point3D.create(100, 0, 0),
                adsk.core.Vector3D.create(-1, 0, 0)))

        self._do_project_point_to_line_test(
            adsk.core.Point3D.create(44732, 0, 1),
            adsk.core.InfiniteLine3D.create(
                adsk.core.Point3D.create(100, 0, 0),
                adsk.core.Vector3D.create(-1, 0, 0)))

    def test_project_point_to_line_angle(self):
        self._do_project_point_to_line_test(
            adsk.core.Point3D.create(1, 1, 0),
            adsk.core.InfiniteLine3D.create(
                adsk.core.Point3D.create(0, 0, 0),
                adsk.core.Vector3D.create(1, 1, 1)))

        self._do_project_point_to_line_test(
            adsk.core.Point3D.create(1, 1, 0),
            adsk.core.InfiniteLine3D.create(
                adsk.core.Point3D.create(64, 2, 337),
                adsk.core.Vector3D.create(1, 1, 1)))

        self._do_project_point_to_line_test(
            adsk.core.Point3D.create(44732, 0, 1),
            adsk.core.InfiniteLine3D.create(
                adsk.core.Point3D.create(100, 100, 100),
                adsk.core.Vector3D.create(1, 1, 1)))


from test_utils import load_tests
def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
