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


class BodyTest(FscadTestCase):
    def validate_test(self):
        pass

    def test_body_after_translate(self):
        box = Box(1, 1, 1)
        body = box.bodies[0]

        self.assertEqual(body.mid().asArray(), (.5, .5, .5))

        box.tx(1)

        self.assertEqual(body.mid().asArray(), (1.5, .5, .5))


def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__],
                                                                #pattern="connected_faces",
                                                                )
    unittest.TextTestRunner(failfast=True).run(test_suite)
