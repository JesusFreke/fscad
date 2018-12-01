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
import importlib
import unittest

test_modules = [
    "difference_test",
    "extrude_test",
    "face_test",
    "fillet_chamfer_test",
    "geometry_2d_test",
    "intersection_test",
    "loft_test",
    "place_test",
    "rotate_test",
    #"scale_test", // temporary disable until scale is fixed
    "translate_test",
    "union_test"
]

def run(context):
    test_suites = []
    for module_name in test_modules:
        module = importlib.import_module(module_name)
        importlib.reload(module)
        module = importlib.import_module(module_name)
        test_suites.append(unittest.defaultTestLoader.loadTestsFromModule(module))

    unittest.TextTestRunner(failfast=True).run(unittest.TestSuite(test_suites))
