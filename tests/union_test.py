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


@test("simple_union_test")
def simple_union_test():
    first = box(1, 1, 1, name="first")
    second = translate(box(1, 1, 1, name="second"), x=1)
    union(first, second, name="union")


@test("overlapping_union_test")
def overlapping_union_test():
    first = box(1, 1, 1, name="first")
    second = translate(box(1, 1, 1, name="second"), x=.5)
    union(first, second, name="union")


@test("disjoint_union_test")
def disjoint_union_test():
    first = box(1, 1, 1, name="first")
    second = translate(box(1, 1, 1, name="second"), x=2)
    union(first, second, name="union")


@test("overlapping_disjoint_union_test")
def overlapping_disjoint_union_test():
    first = box(1, 1, 1, name="first")
    second = translate(box(1, 1, 1, name="second"), x=2)
    first_union = union(first, second, name="first_union")

    third = translate(box(1, 1, 1, name="third"), y=.5)
    fourth = translate(box(1, 1, 1, name="fourth"), x=2, y=.5)
    second_union = union(third, fourth, name="second_union")

    union(first_union, second_union, name="final_union")


@test("joined_overlapping_disjoint_union_test")
def joined_overlapping_disjoint_union_test():
    first = box(1, 1, 1, name="first")
    second = translate(box(1, 1, 1, name="second"), x=2)
    first_union = union(first, second, name="first_union")

    third = translate(box(1, 1, 1, name="third"), y=.5)
    fourth = translate(box(1, 1, 1, name="fourth"), x=2, y=.5)
    second_union = union(third, fourth, name="second_union")

    third_union = union(first_union, second_union, name="third_union")

    fifth = box(3, .1, .1, name="fifth")
    union(fifth, third_union, name="fourth_union")

def run(context):
    run_tests()
