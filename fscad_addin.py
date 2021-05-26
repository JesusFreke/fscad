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
import os.path
import sys

def run(_):
    """Entry point for this Fusion 360 plugin.

    This script can be set up to run as a Fusion 360 plugin on startup, so that the fscad module is automatically
    available for use by other scripts.
    """
    try:
        sys.path.append(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "src"))
        import fscad
        import fscad.fscad
    finally:
        del sys.path[-1]



def stop(_):
    """Callback from Fusion 360 for when this script is being stopped."""
    try:
        to_delete = []
        for key in filter(lambda key: key == "fscad" or key.startswith("fscad."), sys.modules.keys()):
            del sys.modules[key]
    except:
        pass

