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
import unittest

from .fscad import *

#script_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
#script_name = os.path.splitext(os.path.basename(script_path))[0]
#script_dir = os.path.dirname(script_path)

class _FscadWrapperMeta(type):
    def __new__(mcs, class_name, bases, namespace):

        def wrap(func):
            def wrapper(self, *args, **kwargs):
                try:
                    setup_document(self._test_name)
                    func(self)
                    self.validate_test()
                    close_document("expected")
                except:
                    self._close_document = False
                    raise
            return wrapper

        for key, value in list(namespace.items()):
            if callable(value):
                if key.startswith("test_"):
                    namespace[key] = wrap(value)

        return super(_FscadWrapperMeta, mcs).__new__(mcs, class_name, bases, namespace)


class FscadTestCase(unittest.TestCase, metaclass=_FscadWrapperMeta):

    def __init__(self, test_name, results_directory=None, *args, **kwargs):

        if not results_directory:
            self._results_directory = os.path.dirname(os.path.realpath(inspect.getmodule(self).__file__))
        else:
            self._results_directory = results_directory

        self._test_name = test_name[5:]
        self._close_document = True
        super().__init__(test_name, *args, **kwargs)


    def tearDown(self):
        if self._close_document:
            close_document(self._test_name)

    def _compare_occurrence(self, occurrence1, occurrence2, context):
        mycontext = list(context)

        mycontext.append(occurrence1.name)
        self.assertEqual(occurrence1.bRepBodies.count, occurrence2.bRepBodies.count,
                         "%s: Body count doesn't match: %d != %d" % (
                             context, occurrence1.bRepBodies.count, occurrence2.bRepBodies.count))
        bodies1 = list(occurrence1.bRepBodies)
        bodies2 = list(occurrence2.bRepBodies)

        for body1 in bodies1:
            found_match = False
            for body2 in bodies2:
                if equivalent_bodies(body1, body2):
                    bodies2.remove(body2)
                    found_match = True
                    break
            self.assertTrue(found_match, "%s: Couldn't find matching body for %s" % (mycontext, body1.name))

        self.assertEqual(occurrence1.childOccurrences.count, occurrence2.childOccurrences.count,
                         "%s: Child occurrence count doesn't match: %d != %d" % (
                             mycontext, occurrence1.childOccurrences.count, occurrence2.childOccurrences.count))
        for child1 in occurrence1.childOccurrences:
            child2 = occurrence2.childOccurrences.itemByName(child1.name)
            self.assertIsNotNone(child2, "%s: No child corresponding to %s" % (mycontext, child1.name))
            ret = self._compare_occurrence(child1, child2, mycontext)
            if ret is not None:
                return ret

    def validate_test(self):
        occurrences = list(root().occurrences)
        result_occurrence = root().occurrences.addNewComponent(adsk.core.Matrix3D.create())
        result_occurrence.component.name = "actual_result"
        for occurrence in occurrences:
            occurrence.moveToComponent(result_occurrence)

        test_name = self._test_name

        docname = "%s/%s/%s.f3d" % (self._results_directory, self.__class__.__name__, test_name)
        import_options = None
        try:
            import_options = app().importManager.createFusionArchiveImportOptions(docname)
        except RuntimeError:
            pass

        if import_options is None:
            fallback = "%s/%s/%s.f3d" % (self._results_directory, self.__class__.__name__, self._test_name)
            try:
                import_options = app().importManager.createFusionArchiveImportOptions(fallback)
            except RuntimeError:
                pass

        self.assertIsNotNone(import_options, "Couldn't find expected result document: %s" % docname)

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
        self.assertEqual(expected_root.occurrences.count, 1,
                         "Expecting a single occurrence in the root component of the result document")
        expected_root.occurrences.item(0).component.name = "expected_result"

        self._compare_occurrence(expected_root.occurrences.item(0), result_occurrence, [])


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


def unordered_compare(list1, list2):
    list1_copy = list1.copy()

    for item in list2:
        if item in list1_copy:
            list1_copy.remove(item)
        else:
            return False
    if len(list1_copy) != 0:
        return False


def load_tests(loader, standard_tests, pattern):
    test_cases = unittest.TestSuite()

    for standard_test_suite in standard_tests:
        for standard_test in standard_test_suite:
            if pattern is None or pattern == standard_test._test_name:
                test_cases.addTest(standard_test)

    return test_cases
