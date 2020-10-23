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

from . import test_utils
import importlib
importlib.reload(test_utils)
from . import test_utils


class ImportTest(test_utils.FscadTestCase):
    def test_import_dxf(self):
        dxf = import_dxf("%s/%s/import_dxf.dxf" % (self.script_dir, self.__class__.__name__))
        dxf.create_occurrence()


from .test_utils import load_tests
def run(context):
    import sys
    test_suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(failfast=True).run(test_suite)
