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
import math
import traceback
import types
import sys

from typing import Iterable


def _mm(cm_value):
    if cm_value is None:
        return None
    if cm_value is adsk.core.Point3D:
        return adsk.core.Point3D.create(_mm(cm_value.x), _mm(cm_value.y), _mm(cm_value.z))
    return cm_value * 10


def _cm(mm_value):
    if mm_value is None:
        return None
    if mm_value is adsk.core.Point3D:
        return adsk.core.Point3D.create(_cm(mm_value.x), _cm(mm_value.y), _cm(mm_value.z))
    return mm_value / 10


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


def _occurrence_bodies(occurrence: adsk.fusion.Occurrence, only_visible=True, bodies=None)\
        -> Iterable[adsk.fusion.BRepBody]:
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
        body_bounding_box = _oriented_bounding_box_to_bounding_box(
            app().measureManager.getOrientedBoundingBox(body, vector1, vector2))
        if bounding_box is None:
            bounding_box = body_bounding_box
        else:
            bounding_box.combine(body_bounding_box)
    return bounding_box


def _create_component(*bodies, name):
    new_occurrence = root().occurrences.addNewComponent(adsk.core.Matrix3D.create())
    new_occurrence.component.name = name
    base_feature = new_occurrence.component.features.baseFeatures.add()
    base_feature.startEdit()
    for body in bodies:
        new_occurrence.component.bRepBodies.add(body, base_feature)
    base_feature.finishEdit()
    return new_occurrence


def sphere(radius, *, name="Sphere") -> adsk.fusion.Occurrence:
    brep = adsk.fusion.TemporaryBRepManager.get()
    sphere_body = brep.createSphere(adsk.core.Point3D.create(0, 0, 0), _cm(radius))
    return _create_component(sphere_body, name=name)


def cylinder(height, radius, radius2=None, *, name="Cylinder") -> adsk.fusion.Occurrence:
    (height, radius, radius2) = (_cm(height), _cm(radius), _cm(radius2))
    brep = adsk.fusion.TemporaryBRepManager.get()
    cylinder_body = brep.createCylinderOrCone(
        adsk.core.Point3D.create(0, 0, 0),
        radius,
        adsk.core.Point3D.create(0, 0, height),
        radius if radius2 is None else radius2
    )
    return _create_component(cylinder_body, name=name)


def box(x, y, z, *, name="Box"):
    x, y, z = (_cm(x), _cm(y), _cm(z))
    brep = adsk.fusion.TemporaryBRepManager.get()
    box_body = brep.createBox(adsk.core.OrientedBoundingBox3D.create(
        adsk.core.Point3D.create(x/2, y/2, z/2),
        adsk.core.Vector3D.create(1, 0, 0),
        adsk.core.Vector3D.create(0, 1, 0),
        x, y, z))
    return _create_component(box_body, name=name)


def rect(x, y, *, name="Rectangle"):
    sketch = root().sketches.add(root().xYConstructionPlane)
    sketch.sketchCurves.sketchLines.addTwoPointRectangle(
        adsk.core.Point3D.create(0, 0, 0),
        adsk.core.Point3D.create(_cm(x), _cm(y), 0))
    sketch.name = name
    return sketch


def offsetSketch(sketch, offset):
    construction_plane_input = root().constructionPlanes.createInput()
    construction_plane_input.setByOffset(sketch.referencePlane, adsk.core.ValueInput.createByReal(_cm(offset)))
    construction_plane = root().constructionPlanes.add(construction_plane_input)
    construction_plane.isLightBulbOn = False
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
        result_occurrence = component(*occurrences, name=name)
        for occurrence in occurrences:
            occurrence.moveToComponent(result_occurrence)
    else:
        result_occurrence = occurrences[0]

    if vector[0] == 0 and vector[1] == 0 and vector[2] == 0:
        return result_occurrence

    bodies_to_move = adsk.core.ObjectCollection.create()
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
        combine_input = root().features.combineFeatures.createInput(bodies[0], _collection_of(bodies[1:]))
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


def _occurrence_union(occurrences, name):
    bodies = adsk.core.ObjectCollection.create()

    first_body = None
    for occurrence in occurrences:
        for body in _occurrence_bodies(occurrence):
            if first_body is None:
                first_body = body
            else:
                bodies.add(body)

    combine_input = root().features.combineFeatures.createInput(first_body, bodies)
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


def union(*entities, name=None):
    if len(entities) > 0 and isinstance(entities[0], adsk.fusion.Sketch):
        return _sketch_union(entities, name)
    return _occurrence_union(entities, name)


class Joiner(object):
    def __init__(self, join_method, name=None):
        self._entities = []
        self._name = name
        self._join_method = join_method

    def __enter__(self):
        return self

    def __exit__(self, error_type, value, trace):
        if error_type is None:
            occurrence = self._join_method(*self._entities, name=self._name)
            self._occurrence = occurrence

    def __call__(self, entity):
        # TODO: check that the type matches the existing entities
        # also check that the context is still active
        self._entities.append(entity)
        return entity

    def result(self):
        return self._occurrence


def minOf(occurrence):
    return _mm(_get_exact_bounding_box(occurrence).minPoint)


def maxOf(occurrence):
    return _mm(_get_exact_bounding_box(occurrence).maxPoint)


def midOf(occurrence):
    bounding_box = _get_exact_bounding_box(occurrence)
    return _mm(adsk.core.Point3D.create(
        (bounding_box.minPoint.x + bounding_box.maxPoint.x) / 2,
        (bounding_box.minPoint.y + bounding_box.maxPoint.y) / 2,
        (bounding_box.minPoint.z + bounding_box.maxPoint.z) / 2
    ))


def sizeOf(occurrence):
    bounding_box = _get_exact_bounding_box(occurrence)
    return _mm(adsk.core.Point3D.create(
        bounding_box.maxPoint.x - bounding_box.maxPoint.x,
        bounding_box.maxPoint.y - bounding_box.maxPoint.y,
        bounding_box.maxPoint.z - bounding_box.maxPoint.z
    ))


def _get_placement_value(value, coordinate_index):
    if callable(value):
        return value(coordinate_index)
    return _cm(value)


def minAt(value):
    return lambda coordinate_index, bounding_box:\
        _get_placement_value(value, coordinate_index) - bounding_box.minPoint.asArray()[coordinate_index]


def maxAt(value):
    return lambda coordinate_index, bounding_box:\
        _get_placement_value(value, coordinate_index) - bounding_box.maxPoint.asArray()[coordinate_index]


def midAt(value):
    return lambda coordinate_index, bounding_box: \
        _get_placement_value(value, coordinate_index) -\
        (bounding_box.minPoint.asArray()[coordinate_index] + bounding_box.maxPoint.asArray()[coordinate_index]) / 2


def atMin(entity):
    bounding_box = _get_exact_bounding_box(entity)
    return lambda coordinate_index: bounding_box.minPoint.asArray()[coordinate_index]


def atMax(entity):
    bounding_box = _get_exact_bounding_box(entity)
    return lambda coordinate_index: bounding_box.maxPoint.asArray()[coordinate_index]


def atMid(entity):
    bounding_box = _get_exact_bounding_box(entity)
    return lambda coordinate_index:\
        (bounding_box.minPoint.asArray()[coordinate_index] + bounding_box.maxPoint.asArray()[coordinate_index]) / 2


def keep():
    return lambda *_: 0


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


def tx(value, *occurrences, name=None):
    translate((value, 0, 0), *occurrences, name=name)


def ty(value, *occurrences, name=None):
    translate((0, value, 0), *occurrences, name=name)


def tz(value, *occurrences, name=None):
    translate((0, 0, value), *occurrences, name=name)


def rx(value, *occurrences, center=None, name=None):
    rotate((value, 0, 0), *occurrences, center=center, name=name)


def ry(value, *occurrences, center=None, name=None):
    rotate((0, value, 0), *occurrences, center=center, name=name)


def rz(value, *occurrences, center=None, name=None):
    rotate((0, 0, value), *occurrences, center=center, name=name)


def duplicate(func, values, occurrence, keep_original=False):
    for value in values[0:-1]:
        func(value, _duplicate_occurrence(occurrence))
    if keep_original:
        func(values[-1], _duplicate_occurrence(occurrence))
    else:
        func(values[-1], occurrence)


def place(occurrence, x_placement=None, y_placement=None, z_placement=None) -> adsk.fusion.Occurrence:
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

    transform = occurrence.transform
    transform.translation = adsk.core.Vector3D.create(
        x_placement(0, bounding_box),
        y_placement(1, bounding_box),
        z_placement(2, bounding_box))
    occurrence.transform = transform
    design().snapshots.add()
    return occurrence


def run_design(design, message_box_on_error=True, document_name="fSCAD-Preview"):
    """
    Utility method to handle the common setup tasks for a script

    :param design: The function that actually creates the design
    :param message_box_on_error: Set true to pop up a dialog with a stack trace if an error occurs
    :param document_name: The name of the document to create. If a document of the given name already exists, it will
    be forcibly closed and recreated.
    """
    try:
        previewDoc = None  # type: adsk.fusion.FusionDocument
        savedCamera = None
        for document in app().documents:
            if document.name == document_name:
                previewDoc = document
                break
        if previewDoc is not None:
            previewDoc.activate()
            savedCamera = app().activeViewport.camera
            previewDoc.close(False)

        previewDoc = app().documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
        previewDoc.name = document_name
        previewDoc.activate()
        if savedCamera is not None:
            isSmoothTransitionBak = savedCamera.isSmoothTransition
            savedCamera.isSmoothTransition = False
            app().activeViewport.camera = savedCamera
            savedCamera.isSmoothTransition = isSmoothTransitionBak
            app().activeViewport.camera = savedCamera

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


