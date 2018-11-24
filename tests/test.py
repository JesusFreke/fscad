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

import adsk.core
import adsk.fusion
import os
import inspect
import sys
import traceback

from fscad import *

script_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
script_name = os.path.splitext(os.path.basename(script_path))[0]
script_dir = os.path.dirname(script_path)

tests = {}


def test(name):
    def wrapper(func):
        tests[name] = func
        return func
    return wrapper


def validate_test(name):
    occurrences = list(root().occurrences)
    result_occurrence = root().occurrences.addNewComponent(adsk.core.Matrix3D.create())
    result_occurrence.component.name = "actual_result"
    for occurrence in occurrences:
        occurrence.moveToComponent(result_occurrence)
    try:
        import_options = app().importManager.createFusionArchiveImportOptions("%s/%s.f3d" % (script_dir, name))
    except:
        return "Couldn't find expected result document"

    expected_document = None
    for document in app().documents:
        if document.name == "expected":
            expected_document = document
            break
    if expected_document is not None:
        expected_document.close(False)

    expected_document = app().importManager.importToNewDocument(import_options)
    expected_document.name = "expected"
    expected_root = expected_document.design.rootComponent
    if expected_root.occurrences.count > 1:
        return "Expecting a single occurrence in the root component of the result document"
    expected_root.occurrences.item(0).component.name = "expected_result"

    return compare_occurrence(expected_root.occurrences.item(0), result_occurrence, [])


def compare_occurrence(occurrence1, occurrence2, context):
    mycontext = list(context)

    mycontext.append(occurrence1.name)
    if occurrence1.bRepBodies.count != occurrence2.bRepBodies.count:
        return (mycontext,
                "Body count doesn't match: %d != %d" % (occurrence1.bRepBodies.count, occurrence2.bRepBodies.count))
    bodies1 = list(occurrence1.bRepBodies)
    bodies2 = list(occurrence2.bRepBodies)

    for body1 in bodies1:
        found_match = False
        for body2 in bodies2:
            if equivalent_bodies(body1, body2):
                bodies2.remove(body2)
                found_match = True
                break
        if not found_match:
            return (mycontext, "Couldn't find matching body for %s" % body1.name)

    if occurrence1.childOccurrences.count != occurrence2.childOccurrences.count:
           return (mycontext, "Child occurrence count doesn't match: %d != %d" % (
               occurrence1.childOccurrences.count, occurrence2.childOccurrences.count))
    for child1 in occurrence1.childOccurrences:
        child2 = occurrence2.childOccurrences.itemByName(child1.name)
        if child2 is None:
            return (mycontext, "No child corresponding to %s" % child1.name)
        ret = compare_occurrence(child1, child2, mycontext)
        if ret is not None:
            return ret


def equivalent_bodies(body1, body2):
    brep = adsk.fusion.TemporaryBRepManager.get()
    body1_copy = brep.copy(body1)
    body2_copy = brep.copy(body2)
    # If b1 - b2 is empty and b2 - b1 is empty, then the bodies are identical
    brep.booleanOperation(body1_copy, body2, adsk.fusion.BooleanTypes.DifferenceBooleanType)
    if body1_copy.vertices.count != 0:
        return False
    brep.booleanOperation(body2_copy, body1, adsk.fusion.BooleanTypes.DifferenceBooleanType)
    return body2_copy.vertices.count == 0


def close_document(name):
    for document in app().documents:
        if document.name == name:
            document.close(False)
            return


def run_tests():
    for name, test_func in tests.items():
        run_design(test_func, message_box_on_error=False, document_name=name)
        ret = validate_test(name)
        if ret is not None:
            sys.stderr.write("%s failed: %s\n" % (name, ret))
            break
        else:
            close_document(name)
            close_document("expected")





