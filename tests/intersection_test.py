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
import test
import importlib
importlib.reload(test)
from test import *
from fscad import *


@test("simple_intersection_test")
def simple_intersection_test():
    first = box(1, 1, 1, name="first")
    second = translate(box(1, 1, 1, name="second"), x=.5)
    intersection(first, second, name="intersection")


@test("disjoint_intersection_test")
def disjoint_intersection_test():
    first = box(1, 1, 1, name="first")
    second = translate(box(1, 1, 1, name="second"), x=2)
    intersection(first, second, name="intersection")


@test("adjoining_intersection_test")
def adjoining_intersection_test():
    first = box(1, 1, 1, name="first")
    second = translate(box(1, 1, 1, name="second"), x=1)
    intersection(first, second, name="intersection")


@test("complete_intersection_test")
def complete_intersection_test():
    first = box(1, 1, 1, name="first")
    second = box(1, 1, 1, name="second")
    intersection(first, second, name="intersection")


@test("complex_intersection_test")
def complete_intersection_test():
    first = box(1, 1, 1, name="first")
    second = place(box(.5, 10, 10, name="second"),
                   midAt(atMid(first)), midAt(atMid(first)), midAt(atMid(first)))
    difference1 = difference(first, second, name="difference1")

    third = first = box(1, 1, 1, name="third")
    fourth = place(box(10, 10, .5, name="fourth"),
                   midAt(atMid(first)), midAt(atMid(first)), midAt(atMid(first)))
    difference2 = difference(third, fourth, name="difference2")

    intersection1 = intersection(difference1, difference2, name="intersection")


def run(context):
    run_tests()
