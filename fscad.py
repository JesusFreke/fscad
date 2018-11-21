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
import inspect
import math
import traceback
import types
import sys

from typing import Iterable


def _mm(cm_value):
    return cm_value * 10


def _cm(mm_value):
    return mm_value / 10.0


def app():
    return adsk.core.Application.get()


def root() -> adsk.fusion.Component:
    return design().rootComponent


def ui():
    return app().userInterface


def design():
    return adsk.fusion.Design.cast(app().activeProduct)


def user_interface():
    return app().userInterface


def _collection_of(collection):
    object_collection = adsk.core.ObjectCollection.create()
    for obj in collection:
        object_collection.add(obj)
    return object_collection


def _immediate_occurrence_bodies(occurrence, collection):
    for body in occurrence.bRepBodies:
        collection.add(body)


def _occurrence_bodies(occurrence: adsk.fusion.Occurrence, only_visible=True, bodies=None) -> Iterable[adsk.fusion.BRepBody]:
    if bodies is None:
        bodies = adsk.core.ObjectCollection.create()

    _immediate_occurrence_bodies(occurrence, bodies)

    for childOcc in occurrence.childOccurrences:
        if not only_visible or childOcc.isLightBulbOn:
            _occurrence_bodies(childOcc, only_visible, bodies)

    return bodies


def _feature_bodies(feature):
    bodies = adsk.core.ObjectCollection.create()
    for body in feature.bodies:
        bodies.add(body)
    return bodies


def _find_profiles(contained_curve):
    sketch_curve = adsk.fusion.SketchCurve.cast(contained_curve)
    sketch = sketch_curve.parentSketch
    profiles = sketch.profiles
    ret = []

    for profile in profiles:
        loops = profile.profileLoops
        for loop in loops:
            profile_curves = loop.profileCurves
            for profileCurve in profile_curves:
                if profileCurve.sketchEntity == contained_curve:
                    ret.append(profile)
                    break
            else:
                continue
            break
    return ret


def _duplicate_occurrence(occurrence: adsk.fusion.Occurrence, only_visible_bodies=False):
    new_occurrence = root().occurrences.addNewComponent(adsk.core.Matrix3D.create())
    new_occurrence.component.name = occurrence.name
    new_occurrence.isLightBulbOn = occurrence.isLightBulbOn

    if only_visible_bodies:
        for body in _occurrence_bodies(occurrence):
            body.copyToComponent(new_occurrence)
    else:
        for body in occurrence.bRepBodies:
            body.copyToComponent(new_occurrence)
        for childOccurrence in occurrence.childOccurrences:
            _duplicate_occurrence(childOccurrence).moveToComponent(new_occurrence)
    return new_occurrence


def _oriented_bounding_box_to_bounding_box(oriented: adsk.core.OrientedBoundingBox3D):
    return adsk.core.BoundingBox3D.create(
        adsk.core.Point3D.create(
            oriented.centerPoint.x - oriented.length / 2.0,
            oriented.centerPoint.y - oriented.width / 2.0,
            oriented.centerPoint.z - oriented.height / 2.0),
        adsk.core.Point3D.create(
            oriented.centerPoint.x + oriented.length / 2.0,
            oriented.centerPoint.y + oriented.width / 2.0,
            oriented.centerPoint.z + oriented.height / 2.0)
    )


def _get_exact_bounding_box(occurrence):
    vector1 = adsk.core.Vector3D.create(1.0, 0.0, 0.0)
    vector2 = adsk.core.Vector3D.create(0.0, 1.0, 0.0)

    bounding_box = None
    for body in _occurrence_bodies(occurrence):
        body_bounding_box = _oriented_bounding_box_to_bounding_box(app().measureManager.getOrientedBoundingBox(body, vector1, vector2))
        if bounding_box is None:
            bounding_box = body_bounding_box
        else:
            bounding_box.combine(body_bounding_box)
    return bounding_box


def sphere(radius, *, name="Sphere") -> adsk.fusion.Occurrence:
    sketch = root().sketches.add(root().xYConstructionPlane)

    center_point = adsk.core.Point3D.create(0, 0, 0)
    start_point = adsk.core.Point3D.create(_cm(radius), 0, 0)
    arc = sketch.sketchCurves.sketchArcs.addByCenterStartSweep(center_point, start_point, math.pi)
    sketch.sketchCurves.sketchLines.addByTwoPoints(arc.startSketchPoint, arc.endSketchPoint)

    revolves = root().features.revolveFeatures
    revolve_input = revolves.createInput(sketch.profiles.item(0), root().xConstructionAxis,
                                         adsk.fusion.FeatureOperations.NewComponentFeatureOperation)

    angle = adsk.core.ValueInput.createByReal(math.pi)
    revolve_input.isSolid = True
    revolve_input.setAngleExtent(True, angle)

    feature = revolves.add(revolve_input)  # type: adsk.fusion.Feature
    feature.parentComponent.name = name

    return root().allOccurrencesByComponent(feature.parentComponent)[0]


def cylinder(height, radius, radius2=None, *, name="Cylinder") -> adsk.fusion.Occurrence:
    sketch = root().sketches.add(root().xZConstructionPlane)
    origin = adsk.core.Point3D.create(0, 0, 0)
    point1 = adsk.core.Point3D.create(0, _cm(-height), 0)
    point2 = adsk.core.Point3D.create(_cm(radius2) if radius2 is not None else _cm(radius), _cm(-height), 0)
    point3 = adsk.core.Point3D.create(_cm(radius), 0, 0)

    sketch.sketchCurves.sketchLines.addByTwoPoints(origin, point1)
    sketch.sketchCurves.sketchLines.addByTwoPoints(point1, point2)
    sketch.sketchCurves.sketchLines.addByTwoPoints(point2, point3)
    sketch.sketchCurves.sketchLines.addByTwoPoints(point3, origin)

    revolve_input = root().features.revolveFeatures.createInput(
        sketch.profiles.item(0), root().zConstructionAxis, adsk.fusion.FeatureOperations.NewComponentFeatureOperation)
    angle = adsk.core.ValueInput.createByReal(2 * math.pi)
    revolve_input.isSolid = True
    revolve_input.setAngleExtent(True, angle)

    feature = root().features.revolveFeatures.add(revolve_input)
    feature.parentComponent.name = name
    return root().allOccurrencesByComponent(feature.parentComponent)[0]


def box(dimensions, *, name="Box") -> adsk.fusion.Occurrence:
    sketch = root().sketches.add(root().xYConstructionPlane)
    sketch.sketchCurves.sketchLines.addTwoPointRectangle(
        adsk.core.Point3D.create(0, 0, 0),
        adsk.core.Point3D.create(_cm(dimensions[0]), _cm(dimensions[1]), 0))

    extrude_input = root().features.extrudeFeatures.createInput(
        sketch.profiles.item(0),
        adsk.fusion.FeatureOperations.NewComponentFeatureOperation)
    height_value = adsk.core.ValueInput.createByReal(_cm(dimensions[2]))
    height_extent = adsk.fusion.DistanceExtentDefinition.create(height_value)

    extrude_input.setOneSideExtent(height_extent, adsk.fusion.ExtentDirections.PositiveExtentDirection)
    feature = root().features.extrudeFeatures.add(extrude_input)
    feature.parentComponent.name = name
    return root().allOccurrencesByComponent(feature.parentComponent)[0]


def rect(dimensions, *, name="Rectangle"):
    sketch = root().sketches.add(root().xYConstructionPlane)
    sketch.sketchCurves.sketchLines.addTwoPointRectangle(
        adsk.core.Point3D.create(0, 0, 0),
        adsk.core.Point3D.create(_cm(dimensions[0]), _cm(dimensions[1]), 0))
    return sketch


def offsetSketch(sketch, offset):
    construction_plane_input = root().constructionPlanes.createInput()
    construction_plane_input.setByOffset(sketch.referencePlane, adsk.core.ValueInput.createByReal(_cm(offset)))
    construction_plane = root().constructionPlanes.add(construction_plane_input)
    sketch.redefine(construction_plane)


def translateSketch(sketch, translation):
    transform = adsk.core.Matrix3D.create()
    transform.translation = adsk.core.Vector3D.create(_cm(translation[0]), _cm(translation[1]), 0)
    sketch.move(_collection_of(sketch.sketchCurves), transform)
    return sketch


def placeSketch(sketch, origin, normal):
    # TODO: move the sketch around so that the origin of the original sketch is positioned at origin
    normal_vector = adsk.core.Vector3D.create(_cm(normal[0]), _cm(normal[1]), _cm(normal[2]))
    arbitrary_vector = adsk.core.Vector3D.create(0, 0, 1)

    if normal_vector.isParallelTo(arbitrary_vector):
        arbitrary_vector = adsk.core.Vector3D.create(0, 1, 0)

    second_vector = normal_vector.crossProduct(arbitrary_vector)
    third_vector = normal_vector.crossProduct(second_vector)

    sketch2 = root().sketches.add(root().xYConstructionPlane)

    first_point = adsk.core.Point3D.create(_cm(origin[0]), _cm(origin[1]), _cm(origin[2]))
    first_sketch_point = sketch2.sketchPoints.add(first_point)
    second_point = first_point.copy()
    second_point.translateBy(second_vector)
    second_sketch_point = sketch2.sketchPoints.add(second_point)
    third_point = first_point.copy()
    third_point.translateBy(third_vector)
    third_sketch_point = sketch2.sketchPoints.add(third_point)


    construction_plane_input = root().constructionPlanes.createInput()
    construction_plane_input.setByThreePoints(first_sketch_point, second_sketch_point, third_sketch_point)
    construction_plane = root().constructionPlanes.add(construction_plane_input)

    sketch.redefine(construction_plane)

    sketch2.deleteMe()


def loft(*sketches):
    loft_input = root().features.loftFeatures.createInput(adsk.fusion.FeatureOperations.NewComponentFeatureOperation)
    for sketch in sketches:
        if sketch.profiles.count > 1:
            raise ValueError("Sketch %s contains multiple profiles" % sketch.name)
        loft_input.loftSections.add(sketch.profiles.item(0))
    feature = root().features.loftFeatures.add(loft_input)
    return root().allOccurrencesByComponent(feature.parentComponent)[0]


def _do_intersection(target_occurrence, tool_occurrence):

    tool_bodies = adsk.core.ObjectCollection.create()  # type: adsk.core.ObjectCollection
    for tool_body in _occurrence_bodies(tool_occurrence):
        tool_bodies.add(tool_body)

    for target_body in _occurrence_bodies(target_occurrence):
        combine_input = target_occurrence.component.features.combineFeatures.createInput(target_body, tool_bodies)
        combine_input.operation = adsk.fusion.FeatureOperations.IntersectFeatureOperation
        combine_input.isKeepToolBodies = True
        target_occurrence.component.features.combineFeatures.add(combine_input)


def intersection(*occurrences, name="Intersection") -> adsk.fusion.Occurrence:
    intersection_occurrence = root().occurrences.addNewComponent(adsk.core.Matrix3D.create())
    intersection_component = intersection_occurrence.component  # type: adsk.fusion.Component
    intersection_component.name = name

    starting_occurrence = occurrences[0]

    for body in _occurrence_bodies(starting_occurrence):
        body.copyToComponent(intersection_occurrence)

    for tool_occurrence in occurrences[1:]:
        _do_intersection(intersection_occurrence, tool_occurrence)

    for occurrence in occurrences:
        occurrence.moveToComponent(intersection_occurrence)
        occurrence.isLightBulbOn = False
    return intersection_occurrence


def _do_difference(target_occurrence, tool_occurrence):
    tool_bodies = adsk.core.ObjectCollection.create()  # type: adsk.core.ObjectCollection
    for tool_body in _occurrence_bodies(tool_occurrence):
        tool_bodies.add(tool_body)

    for target_body in _occurrence_bodies(target_occurrence):
        combine_input = target_occurrence.component.features.combineFeatures.createInput(target_body, tool_bodies)
        combine_input.operation = adsk.fusion.FeatureOperations.CutFeatureOperation
        combine_input.isKeepToolBodies = True
        target_occurrence.component.features.combineFeatures.add(combine_input)


def difference(*occurrences, name=None) -> adsk.fusion.Occurrence:
    base_occurrence = occurrences[0]

    difference_occurrence = _duplicate_occurrence(base_occurrence, True)
    difference_occurrence.component.name = name or base_occurrence.name

    for tool_occurrence in occurrences[1:]:
        _do_difference(difference_occurrence, tool_occurrence)

    for occurrence in occurrences:
        occurrence.moveToComponent(difference_occurrence)
        occurrence.isLightBulbOn = False
    return difference_occurrence


def translate(vector, *occurrences, name="Translate"):
    if len(occurrences) > 1:
        print("here1")
        result_occurrence = component(*occurrences, name=name)
        for occurrence in occurrences:
            occurrence.moveToComponent(result_occurrence)
    else:
        print("here2")
        result_occurrence = occurrences[0]

    if vector[0] == 0 and vector[1] == 0 and vector[2] == 0:
        return result_occurrence

    bodies_to_move = adsk.core.ObjectCollection.create()
    bodies = _occurrence_bodies(result_occurrence, only_visible=False)
    for body in _occurrence_bodies(result_occurrence, only_visible=False):
        bodies_to_move.add(body)

    transform = adsk.core.Matrix3D.create()
    transform.translation = adsk.core.Vector3D.create(_cm(vector[0]), _cm(vector[1]), _cm(vector[2]))
    move_input = result_occurrence.component.features.moveFeatures.createInput(bodies_to_move, transform)
    result_occurrence.component.features.moveFeatures.add(move_input)
    return result_occurrence


def rotate(angles, *occurrences, center=None, name="Rotate"):
    if center is None:
        center = adsk.core.Point3D.create(0, 0, 0)
    else:
        center = adsk.core.Point3D.create(_cm(center[0]), _cm(center[1]), _cm(center[2]))

    if len(occurrences) > 1:
        result_occurrence = component(*occurrences, name=name)
    else:
        result_occurrence = occurrences[0]

    if angles[0] == 0 and angles[1] == 0 and angles[2] == 0:
        return result_occurrence

    bodies_to_rotate = adsk.core.ObjectCollection.create()
    for body in _occurrence_bodies(result_occurrence):
        bodies_to_rotate.add(body)

    transform1 = adsk.core.Matrix3D.create()
    transform1.setToRotation(math.radians(angles[0]), adsk.core.Vector3D.create(1, 0, 0), center)
    transform2 = adsk.core.Matrix3D.create()
    transform2.setToRotation(math.radians(angles[1]), adsk.core.Vector3D.create(0, 1, 0), center)
    transform3 = adsk.core.Matrix3D.create()
    transform3.setToRotation(math.radians(angles[2]), adsk.core.Vector3D.create(0, 0, 1), center)

    transform2.invert()
    transform3.invert()

    transform1.transformBy(transform2)
    transform1.transformBy(transform3)

    move_input = result_occurrence.component.features.moveFeatures.createInput(bodies_to_rotate, transform1)
    result_occurrence.component.features.moveFeatures.add(move_input)
    return result_occurrence


def component(*occurrences, name="Component") -> adsk.fusion.Occurrence:
    new_occurrence = root().occurrences.addNewComponent(adsk.core.Matrix3D.create())  # type: adsk.fusion.Occurrence
    new_component = new_occurrence.component  # type: adsk.fusion.Component
    new_component.name = name

    for occurrence in occurrences:
        occurrence.moveToComponent(new_occurrence)
    return new_occurrence


def _extrude(sketch, amount):
    profiles = _collection_of(sketch.profiles)
    extrude_input = root().features.extrudeFeatures.createInput(
        profiles, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)

    height_value = adsk.core.ValueInput.createByReal(_cm(amount))
    height_extent = adsk.fusion.DistanceExtentDefinition.create(height_value)
    extrude_input.setOneSideExtent(height_extent, adsk.fusion.ExtentDirections.PositiveExtentDirection)

    return root().features.extrudeFeatures.add(extrude_input)


def _sketch_union(sketches, name):
    intermediate_features = []

    # TODO: project any sketches or occurrences onto the first sketches' plane
    bodies = []
    result_bodies = []
    for sketch in sketches:
        extrude_feature = _extrude(sketch, 1)
        intermediate_features.append(extrude_feature)
        for body in extrude_feature.bodies:
            bodies.append(body)

    if len(bodies) > 1:
        combine_input = root().features.combineFeatures.createInput(bodies[0], _collection_of(bodies[1:]))  # type: adsk.fusion.CombineFeatureInput
        combine_input.operation = adsk.fusion.FeatureOperations.JoinFeatureOperation
        combine_input.isKeepToolBodies = True
        combine_input.isNewComponent = True
        combine_feature = root().features.combineFeatures.add(combine_input)
        for body in combine_feature.parentComponent.bRepBodies:
            result_bodies.append(body)
    else:
        result_bodies.append(bodies[0])

    temp_sketch = combine_feature.parentComponent.sketches.add(sketches[0].referencePlane)
    new_sketch = root().sketches.add(sketches[0].referencePlane)

    temp_sketch.intersectWithSketchPlane(result_bodies)
    temp_sketch.copy(_collection_of(temp_sketch.sketchCurves), adsk.core.Matrix3D.create(), new_sketch)

    combine_feature.deleteMe()
    root().allOccurrencesByComponent(combine_feature.parentComponent)[0].deleteMe()
    for feature in intermediate_features:
        feature.dissolve()
    for body in bodies:
        body.deleteMe()

    if name is not None:
        new_sketch.name = name

    return new_sketch


def union(*occurrences, name=None) -> adsk.fusion.Occurrence:
    if len(occurrences) > 0 and isinstance(occurrences[0], adsk.fusion.Sketch):
        return _sketch_union(occurrences, name)

    bodies = adsk.core.ObjectCollection.create()

    first_body = None
    for occurrence in occurrences:
        for body in _occurrence_bodies(occurrence):
            if first_body is None:
                first_body = body
            else:
                bodies.add(body)

    combine_input = root().features.combineFeatures.createInput(first_body, bodies)  # type: adsk.fusion.CombineFeatureInput
    combine_input.operation = adsk.fusion.FeatureOperations.JoinFeatureOperation
    combine_input.isKeepToolBodies = True
    combine_input.isNewComponent = True
    feature = root().features.combineFeatures.add(combine_input)

    union_occurrence = root().allOccurrencesByComponent(feature.parentComponent)[0]

    for occurrence in occurrences:
        occurrence.moveToComponent(union_occurrence)
        occurrence.isLightBulbOn = False
    if name is not None:
        union_occurrence.component.name = name
    return union_occurrence


def _old_place(occurrence, x=None, y=None, z=None):
    bounding_box = _get_exact_bounding_box(occurrence)

    translation_x = 0
    if x is not None:
        if x[0] == "min":
            translation_x = x[1] - _mm(bounding_box.minPoint.x)
        elif x[0] == "max":
            translation_x = x[1] - _mm(bounding_box.maxPoint.x)
        elif x[0] == "mid":
            translation_x = x[1] - _mm(bounding_box.maxPoint.x + bounding_box.minPoint.x) / 2.0
        elif x[0] == "keep":
            translation_x = 0
        else:
            raise ValueError("invalid x alignment type: %s" % x[0])

    translation_y = 0
    if y is not None:
        if y[0] == "min":
            translation_y = y[1] - _mm(bounding_box.minPoint.y)
        elif y[0] == "max":
            translation_y = y[1] - _mm(bounding_box.maxPoint.y)
        elif y[0] == "mid":
            translation_y = y[1] - _mm(bounding_box.maxPoint.y + bounding_box.minPoint.y) / 2.0
        elif y[0] == "keep":
            translation_y = 0
        else:
            raise ValueError("invalid y alignment type: %s" % y[0])

    translation_z = 0
    if z is not None:
        if z[0] == "min":
            translation_z = z[1] - _mm(bounding_box.minPoint.z)
        elif z[0] == "max":
            translation_z = z[1] - _mm(bounding_box.maxPoint.z)
        elif z[0] == "mid":
            translation_z = z[1] - _mm(bounding_box.maxPoint.z + bounding_box.minPoint.z) / 2.0
        elif z[0] == "keep":
            translation_z = 0
        else:
            raise ValueError("invalid z alignment type: %s" % z[0])

    return translate((translation_x, translation_y, translation_z), occurrence)

def minX(occurrence: adsk.fusion.Occurrence):
    return _mm(_get_exact_bounding_box(occurrence).minPoint.x)


def maxX(occurrence: adsk.fusion.Occurrence):
    return _mm(_get_exact_bounding_box(occurrence).maxPoint.x)


def midX(occurrence: adsk.fusion.Occurrence):
    bounding_box = _get_exact_bounding_box(occurrence)
    return _mm(bounding_box.maxPoint.x + bounding_box.minPoint.x)/2.0


def sizeX(occurrence: adsk.fusion.Occurrence):
    bounding_box = _get_exact_bounding_box(occurrence)
    return _mm(bounding_box.maxPoint.x - bounding_box.minPoint.x)


def minY(occurrence: adsk.fusion.Occurrence):
    return _mm(_get_exact_bounding_box(occurrence).minPoint.y)


def maxY(occurrence: adsk.fusion.Occurrence):
    return _mm(_get_exact_bounding_box(occurrence).maxPoint.y)


def midY(occurrence: adsk.fusion.Occurrence):
    bounding_box = _get_exact_bounding_box(occurrence)
    return _mm(bounding_box.maxPoint.y + bounding_box.minPoint.y)/2.0


def sizeY(occurrence: adsk.fusion.Occurrence):
    bounding_box = _get_exact_bounding_box(occurrence)
    return _mm(bounding_box.maxPoint.y - bounding_box.minPoint.y)


def minZ(occurrence: adsk.fusion.Occurrence):
    return _mm(_get_exact_bounding_box(occurrence).minPoint.z)


def maxZ(occurrence: adsk.fusion.Occurrence):
    return _mm(_get_exact_bounding_box(occurrence).maxPoint.z)


def midZ(occurrence: adsk.fusion.Occurrence):
    bounding_box = _get_exact_bounding_box(occurrence)
    return _mm(bounding_box.maxPoint.z + bounding_box.minPoint.z)/2.0


def sizeZ(occurrence: adsk.fusion.Occurrence):
    bounding_box = _get_exact_bounding_box(occurrence)
    return _mm(bounding_box.maxPoint.z - bounding_box.minPoint.z)


def minAt(value):
    return "min", value


def maxAt(value):
    return "max", value


def midAt(value):
    return "mid", value


def keep():
    return "keep",


def touching(anchor_occurrence, target_occurrence):
    measure_result = app().measureManager.measureMinimumDistance(target_occurrence, anchor_occurrence)

    translate((
        _mm(measure_result.positionTwo.x - measure_result.positionOne.x),
        _mm(measure_result.positionTwo.y - measure_result.positionOne.y),
        _mm(measure_result.positionTwo.z - measure_result.positionOne.z)),
        target_occurrence)


def distance_between(occurrence1, occurrence2):
    measure_result = app().measureManager.measureMinimumDistance(occurrence1, occurrence2)
    return math.sqrt(
        math.pow(_mm(measure_result.positionTwo.x - measure_result.positionOne.x), 2) +
        math.pow(_mm(measure_result.positionTwo.y - measure_result.positionOne.y), 2) +
        math.pow(_mm(measure_result.positionTwo.z - measure_result.positionOne.z), 2))


def tx(value, *occurrences):
    translate((value, 0, 0), *occurrences)


def ty(value, *occurrences):
    translate((0, value, 0), *occurrences)


def tz(value, *occurrences):
    translate((0, 0, value), *occurrences)


def rx(value, *occurrences):
    rotate((value, 0, 0), *occurrences)


def ry(value, *occurrences):
    rotate((0, value, 0), *occurrences)


def rz(value, *occurrences):
    rotate((0, 0, value), *occurrences)


def rotate_duplicate(angles, occurrence, center=None):
    ret = []
    for angle in angles[0:-1]:
        ret.append(rotate((0, 0, angle), _duplicate_occurrence(occurrence), center=center))
    ret.append(rotate((0, 0, angles[-1]), occurrence, center=center))
    return ret


def place(occurrence, x=None, y=None, z=None):
    transform = occurrence.transform
    transform.translation = adsk.core.Vector3D.create(0, 0, 0)
    occurrence.transform = transform

    vector1 = adsk.core.Vector3D.create(1.0, 0.0, 0.0)
    vector2 = adsk.core.Vector3D.create(0.0, 1.0, 0.0)

    bounding_box = adsk.core.BoundingBox3D.create(adsk.core.Point3D.create(0, 0, 0), adsk.core.Point3D.create(0, 0, 0))
    for body in _occurrence_bodies(occurrence):
        bounding_box.combine(
            _oriented_bounding_box_to_bounding_box(
                app().measureManager.getOrientedBoundingBox(body, vector1, vector2)))

    translation_x = 0
    if x is not None:
        if x[0] == "min":
            translation_x = x[1] - _mm(bounding_box.minPoint.x)
        elif x[0] == "max":
            translation_x = x[1] - _mm(bounding_box.maxPoint.x)
        elif x[0] == "mid":
            translation_x = x[1] - _mm(bounding_box.maxPoint.x + bounding_box.minPoint.x) / 2.0
        elif x[0] == "keep":
            translation_x = 0
        else:
            raise ValueError("invalid x alignment type: %s" % x[0])

    translation_y = 0
    if y is not None:
        if y[0] == "min":
            translation_y = y[1] - _mm(bounding_box.minPoint.y)
        elif y[0] == "max":
            translation_y = y[1] - _mm(bounding_box.maxPoint.y)
        elif y[0] == "mid":
            translation_y = y[1] - _mm(bounding_box.maxPoint.y + bounding_box.minPoint.y) / 2.0
        elif y[0] == "keep":
            translation_y = 0
        else:
            raise ValueError("invalid y alignment type: %s" % y[0])

    translation_z = 0
    if z is not None:
        if z[0] == "min":
            translation_z = z[1] - _mm(bounding_box.minPoint.z)
        elif z[0] == "max":
            translation_z = z[1] - _mm(bounding_box.maxPoint.z)
        elif z[0] == "mid":
            translation_z = z[1] - _mm(bounding_box.maxPoint.z + bounding_box.minPoint.z) / 2.0
        elif z[0] == "keep":
            translation_z = 0
        else:
            raise ValueError("invalid z alignment type: %s" % z[0])

    transform = occurrence.transform
    transform.translation = adsk.core.Vector3D.create(_cm(translation_x), _cm(translation_y), _cm(translation_z))
    occurrence.transform = transform
    design().snapshots.add()
    return occurrence

def run_design(design, message_box_on_error=True, document_name="fSCAD-Preview"):
    try:
        previewDoc = None
        for document in app().documents:
            if document.name == document_name:
                previewDoc = document
                break
        if previewDoc is not None:
            previewDoc.close(False)

        previewDoc = app().documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
        previewDoc.name = document_name
        previewDoc.activate()

        design()
    except:
        print(traceback.format_exc())
        if message_box_on_error:
            ui = user_interface()
            if ui:
                ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))


def run(context):
    fscad = types.ModuleType("fscad")
    sys.modules['fscad'] = fscad

    for key, value in globals().items():
        if not callable(value):
            continue
        if key.startswith('_'):
            continue
        if key == "run" or key == "stop":
            continue
        fscad.__setattr__(key, value)


def stop(context):
    del sys.modules['fscad']


