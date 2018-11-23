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


def _convert_units(value, convert):
    if value is None:
        return None
    if isinstance(value, adsk.core.Point3D):
        return adsk.core.Point3D.create(convert(value.x), convert(value.y), convert(value.z))
    if isinstance(value, adsk.core.Vector3D):
        return adsk.core.Vector3D.create(convert(value.x), convert(value.y), convert(value.z))
    if isinstance(value, tuple):
        return tuple(map(convert, value))
    if isinstance(value, list):
        return list(map(convert, value))
    return convert(value)


def _mm(cm_value):
    return _convert_units(cm_value, lambda value: value * 10)


def _cm(mm_value):
    return _convert_units(mm_value, lambda value: value / 10)


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


def _occurrence_bodies(occurrence: adsk.fusion.Occurrence, bodies=None)\
        -> Iterable[adsk.fusion.BRepBody]:
    if bodies is None:
        bodies = adsk.core.ObjectCollection.create()
    for body in occurrence.bRepBodies:
        bodies.add(body)
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

    if only_visible_bodies:
        for body in _occurrence_bodies(occurrence):
            body.copyToComponent(new_occurrence)
    else:
        for body in occurrence.bRepBodies:
            body.copyToComponent(new_occurrence)
        for childOccurrence in occurrence.childOccurrences:
            _duplicate_occurrence(childOccurrence).moveToComponent(new_occurrence)

    if not occurrence.isLightBulbOn:
        _hide_occurrence(new_occurrence)
    return new_occurrence


def _hide_occurrence(occurrence):
    occurrence.isLightBulbOn = False
    occurrence.component.isBodiesFolderLightBulbOn = False


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


def _sketch_to_wire_body(sketch):
    brep = adsk.fusion.TemporaryBRepManager.get()

    curves = []
    for curve in sketch.sketchCurves:
        curves.append(curve.geometry)

    wire, edge_map = brep.createWireFromCurves(curves)
    return wire


def _get_exact_bounding_box(entity):
    vector1 = adsk.core.Vector3D.create(1.0, 0.0, 0.0)
    vector2 = adsk.core.Vector3D.create(0.0, 1.0, 0.0)

    if isinstance(entity, adsk.fusion.Sketch):
        bodies = [_sketch_to_wire_body(entity)]
    else:
        bodies = _occurrence_bodies(entity)

    bounding_box = None
    for body in bodies:
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
    for body in bodies:
        new_occurrence.component.bRepBodies.add(body)
    return new_occurrence


def sphere(radius, *, name="Sphere") -> adsk.fusion.Occurrence:
    brep = adsk.fusion.TemporaryBRepManager.get()
    sphere_body = brep.createSphere(adsk.core.Point3D.create(0, 0, 0), _cm(radius))
    return _create_component(sphere_body, name=name)


def cylinder(height, radius, radius2=None, *, name="Cylinder") -> adsk.fusion.Occurrence:
    (height, radius, radius2) = _cm((height, radius, radius2))
    brep = adsk.fusion.TemporaryBRepManager.get()
    cylinder_body = brep.createCylinderOrCone(
        adsk.core.Point3D.create(0, 0, 0),
        radius,
        adsk.core.Point3D.create(0, 0, height),
        radius if radius2 is None else radius2
    )
    return _create_component(cylinder_body, name=name)


def box(x, y, z, *, name="Box"):
    x, y, z = _cm((x, y, z))
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
        _cm(adsk.core.Point3D.create(x, y, 0)))
    sketch.name = name
    return sketch


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
        _hide_occurrence(occurrence)
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
        _hide_occurrence(occurrence)
    return difference_occurrence


def _translate_occurrence(occurrence, x, y, z):
    if x == 0 and y == 0 and z == 0:
        return occurrence

    new_occurrence = root().occurrences.addNewComponent(adsk.core.Matrix3D.create())

    bodies_to_move = adsk.core.ObjectCollection.create()
    for body in _occurrence_bodies(occurrence):
        bodies_to_move.add(body.copyToComponent(new_occurrence))

    transform = adsk.core.Matrix3D.create()
    transform.translation = _cm(adsk.core.Vector3D.create(x, y, z))
    move_input = new_occurrence.component.features.moveFeatures.createInput(bodies_to_move, transform)
    new_occurrence.component.features.moveFeatures.add(move_input)
    occurrence.moveToComponent(new_occurrence)
    _hide_occurrence(occurrence)
    new_occurrence.component.name = occurrence.name
    return new_occurrence


def _translate_sketch(sketch, x, y, z):
    if z:
        construction_plane_input = root().constructionPlanes.createInput()
        construction_plane_input.setByOffset(sketch.referencePlane, adsk.core.ValueInput.createByReal(_cm(z)))
        construction_plane = root().constructionPlanes.add(construction_plane_input)
        construction_plane.isLightBulbOn = False
        sketch.redefine(construction_plane)
    matrix = adsk.core.Matrix3D.create()
    matrix.translation = _cm(adsk.core.Vector3D.create(x, y, 0))
    sketch.move(_collection_of(sketch.sketchCurves), matrix)
    return sketch


def translate(entity, x=0, y=0, z=0):
    if isinstance(entity, adsk.fusion.Sketch):
        _translate_sketch(entity, x, y, z)
    else:
        _translate_occurrence(entity, x, y, z)


def rotate(occurrence, x, y, z, center=None):
    if x == 0 and y == 0 and z == 0:
        return occurrence

    new_occurrence = root().occurrences.addNewComponent(adsk.core.Matrix3D.create())

    if center is None:
        center = adsk.core.Point3D.create(0, 0, 0)
    else:
        center = _cm(adsk.core.Point3D.create(*center))

    bodies_to_rotate = adsk.core.ObjectCollection.create()
    for body in _occurrence_bodies(occurrence):
        bodies_to_rotate.add(body.copyToComponent(new_occurrence))

    transform1 = adsk.core.Matrix3D.create()
    transform1.setToRotation(math.radians(x), adsk.core.Vector3D.create(1, 0, 0), center)
    transform2 = adsk.core.Matrix3D.create()
    transform2.setToRotation(math.radians(y), adsk.core.Vector3D.create(0, 1, 0), center)
    transform3 = adsk.core.Matrix3D.create()
    transform3.setToRotation(math.radians(z), adsk.core.Vector3D.create(0, 0, 1), center)

    transform2.invert()
    transform3.invert()

    transform1.transformBy(transform2)
    transform1.transformBy(transform3)

    move_input = new_occurrence.component.features.moveFeatures.createInput(bodies_to_rotate, transform1)
    new_occurrence.component.features.moveFeatures.add(move_input)
    occurrence.moveToComponent(new_occurrence)
    _hide_occurrence(occurrence)
    new_occurrence.component.name = occurrence.name
    return new_occurrence


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
        _hide_occurrence(occurrence)
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
        return _cm(value(coordinate_index))
    return _cm(value)


def minAt(value):
    return lambda coordinate_index, bounding_box: _mm(
        _get_placement_value(value, coordinate_index) - bounding_box.minPoint.asArray()[coordinate_index])


def maxAt(value):
    return lambda coordinate_index, bounding_box: _mm(
        _get_placement_value(value, coordinate_index) - bounding_box.maxPoint.asArray()[coordinate_index])


def midAt(value):
    return lambda coordinate_index, bounding_box: _mm(
        _get_placement_value(value, coordinate_index) -
        (bounding_box.minPoint.asArray()[coordinate_index] + bounding_box.maxPoint.asArray()[coordinate_index]) / 2)


def atMin(entity):
    bounding_box = _get_exact_bounding_box(entity)
    return lambda coordinate_index: _mm(bounding_box.minPoint.asArray()[coordinate_index])


def atMax(entity):
    bounding_box = _get_exact_bounding_box(entity)
    return lambda coordinate_index: _mm(bounding_box.maxPoint.asArray()[coordinate_index])


def atMid(entity):
    bounding_box = _get_exact_bounding_box(entity)
    return lambda coordinate_index: _mm(
        (bounding_box.minPoint.asArray()[coordinate_index] + bounding_box.maxPoint.asArray()[coordinate_index]) / 2)


def keep():
    return lambda *_: 0


def touching(anchor_occurrence, target_occurrence):
    measure_result = app().measureManager.measureMinimumDistance(target_occurrence, anchor_occurrence)

    translate(target_occurrence,  *_mm((
        _mm(measure_result.positionTwo.x - measure_result.positionOne.x),
        _mm(measure_result.positionTwo.y - measure_result.positionOne.y),
        _mm(measure_result.positionTwo.z - measure_result.positionOne.z))))


def distance_between(occurrence1, occurrence2):
    measure_result = app().measureManager.measureMinimumDistance(occurrence1, occurrence2)
    return math.sqrt(
        math.pow(_mm(measure_result.positionTwo.x - measure_result.positionOne.x), 2) +
        math.pow(_mm(measure_result.positionTwo.y - measure_result.positionOne.y), 2) +
        math.pow(_mm(measure_result.positionTwo.z - measure_result.positionOne.z), 2))


def tx(occurrence, translation):
    return translate(occurrence, translation, 0, 0)


def ty(occurrence, translation):
    return translate(occurrence, 0, translation, 0)


def tz(occurrence, translation):
    return translate(occurrence, 0, 0, translation)


def rx(occurrence, angle, center=None):
    return rotate(occurrence, angle, 0, 0, center=center)


def ry(occurrence, angle, center=None):
    return rotate(occurrence, 0, angle, 0, center=center)


def rz(occurrence, angle, center=None):
    return rotate(occurrence, 0, 0, angle, center=center)


def duplicate(func, values, occurrence, keep_original=True):
    result_occurrence = root().occurrences.addNewComponent(adsk.core.Matrix3D.create())
    result_occurrence.component.name = occurrence.name

    def handle_result(result):
        for body in result.bRepBodies:
            body.copyToComponent(result_occurrence)
        result.deleteMe()

    for value in values:
        handle_result(func(_duplicate_occurrence(occurrence), value))

    if keep_original:
        handle_result(_duplicate_occurrence(occurrence))
    occurrence.moveToComponent(result_occurrence)
    _hide_occurrence(occurrence)
    return result_occurrence


def _place_occurrence(occurrence, x_placement=None, y_placement=None, z_placement=None) -> adsk.fusion.Occurrence:
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
    transform.translation = _cm(adsk.core.Vector3D.create(
        x_placement(0, bounding_box),
        y_placement(1, bounding_box),
        z_placement(2, bounding_box)))
    occurrence.transform = transform
    return occurrence


def _place_sketch(sketch, x_placement=None, y_placement=None, z_placement=None) -> adsk.fusion.Occurrence:
    bounding_box = _get_exact_bounding_box(sketch)
    translate((
        x_placement(0, bounding_box),
        y_placement(1, bounding_box),
        z_placement(2, bounding_box)),
        sketch
    )
    return sketch


def place(entity, x_placement=None, y_placement=None, z_placement=None) -> adsk.fusion.Occurrence:
    if isinstance(entity, adsk.fusion.Sketch):
        return _place_sketch(entity, x_placement, y_placement, z_placement)
    else:
        return _place_occurrence(entity, x_placement, y_placement, z_placement)


def run_design(design_func, message_box_on_error=True, document_name="fSCAD-Preview"):
    """
    Utility method to handle the common setup tasks for a script

    :param design_func: The function that actually creates the design
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
        design().designType = adsk.fusion.DesignTypes.DirectDesignType
        if savedCamera is not None:
            isSmoothTransitionBak = savedCamera.isSmoothTransition
            savedCamera.isSmoothTransition = False
            app().activeViewport.camera = savedCamera
            savedCamera.isSmoothTransition = isSmoothTransitionBak
            app().activeViewport.camera = savedCamera

        design_func()
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


