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

# note: load_tests is required for the "pattern" test filtering functionality in loadTestsFromModule in run()
from fscad.test_utils import FscadTestCase, load_tests
from fscad.fscad import *


class RawThreadTest(FscadTestCase):
    def test_square_thread(self):
        points = [
            (0, 0),
            (.5, 0),
            (.5, .5),
            (0, .5),
        ]
        threads = RawThreads(
            inner_radius=5,
            thread_profile=points,
            pitch=1,
            turns=2)
        threads.create_occurrence()
        threads.start_face.make_component(name="StartFace").create_occurrence()
        threads.end_face.make_component(name="EndFace").create_occurrence()

    def test_partial_square_thread(self):
        points = [
            (0, 0),
            (.5, 0),
            (.5, .5),
            (0, .5),
        ]
        threads = RawThreads(
            inner_radius=5,
            thread_profile=points,
            pitch=1,
            turns=.5)
        threads.create_occurrence()
        threads.start_face.make_component(name="StartFace").create_occurrence()
        threads.end_face.make_component(name="EndFace").create_occurrence()

    def test_trapezoidal_thread(self):
        points = [
            (0, 0),
            (.25, .25),
            (.25, .5),
            (0, .75),
        ]
        threads = RawThreads(
            inner_radius=5,
            thread_profile=points,
            pitch=1,
            turns=2)
        threads.create_occurrence()
        threads.start_face.make_component(name="StartFace").create_occurrence()
        threads.end_face.make_component(name="EndFace").create_occurrence()

    def test_dense_trapezoidal_thread(self):
        points = [
            (0, 0),
            (.25, .25),
            (.25, .5),
            (0, .75),
        ]
        threads = RawThreads(
            inner_radius=5,
            thread_profile=points,
            pitch=.75,
            turns=2)
        threads.create_occurrence()
        threads.start_face.make_component(name="StartFace").create_occurrence()
        threads.end_face.make_component(name="EndFace").create_occurrence()


def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__],
                                                                # pattern="square_thread",
                                                                )
    unittest.TextTestRunner(failfast=True).run(test_suite)
