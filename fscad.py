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
import uuid

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


def _get_parent_component(occurrence):
    if occurrence.assemblyContext is None:
        return root()
    return occurrence.assemblyContext.component


def _occurrence_bodies(occurrence: adsk.fusion.Occurrence):
    bodies = []
    for body in occurrence.bRepBodies:
        bodies.append(body)
    for child in occurrence.childOccurrences:
        if child.isLightBulbOn:
            bodies.extend(_occurrence_bodies(child))
    return bodies


def _check_2D(occurrence):
    has2D = False
    has3D = False
    for body in _occurrence_bodies(occurrence):
        if body.isSolid:
            has3D = True
        else:
            has2D = True
        if has2D and has3D:
            raise ValueError("Occurrence %s contains both 2D and 3D geometry" % occurrence.name)
    return has2D


def _check_coplanarity(plane1, plane2):
    if plane1 is None or plane2 is None:
        return plane1 or plane2
    if not plane1.isCoPlanarTo(plane2):
        raise ValueError("Cannot perform operation on non-coplanar 2D geometery")
    return plane1 or plane2


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

    bodies = _occurrence_bodies(occurrence)

    bounding_box = None
    for body in bodies:
        body_bounding_box = _oriented_bounding_box_to_bounding_box(
            app().measureManager.getOrientedBoundingBox(body, vector1, vector2))
        if bounding_box is None:
            bounding_box = body_bounding_box
        else:
            bounding_box.combine(body_bounding_box)
    return bounding_box


def _create_component(parent_component, *bodies, name):
    new_occurrence = parent_component.occurrences.addNewComponent(adsk.core.Matrix3D.create())
    new_occurrence.component.name = name
    base_feature = new_occurrence.component.features.baseFeatures.add()
    base_feature.startEdit()
    for body in bodies:
        new_occurrence.component.bRepBodies.add(body, base_feature)
    base_feature.finishEdit()
    return new_occurrence


def _mark_face(face, face_name):
    face_uuid = uuid.uuid4()
    face.attributes.add("fscad", "id", str(face_uuid))
    face.attributes.add("fscad", str(face_uuid), str(face_uuid))
    face.body.attributes.add("fscad", face_name, str(face_uuid))


def sphere(radius, *, name="Sphere") -> adsk.fusion.Occurrence:
    brep = adsk.fusion.TemporaryBRepManager.get()
    sphere_body = brep.createSphere(adsk.core.Point3D.create(0, 0, 0), _cm(radius))
    occurrence = _create_component(root(), sphere_body, name=name)
    _mark_face(occurrence.bRepBodies.item(0).faces.item(0), "surface")


def cylinder(height, radius, radius2=None, *, name="Cylinder") -> adsk.fusion.Occurrence:
    (height, radius, radius2) = _cm((height, radius, radius2))
    brep = adsk.fusion.TemporaryBRepManager.get()
    cylinder_body = brep.createCylinderOrCone(
        adsk.core.Point3D.create(0, 0, 0),
        radius,
        adsk.core.Point3D.create(0, 0, height),
        radius if radius2 is None else radius2
    )
    occurrence = _create_component(root(), cylinder_body, name=name)
    for face in occurrence.bRepBodies.item(0).faces:
        if face.geometry.surfaceType == adsk.core.SurfaceTypes.CylinderSurfaceType or \
                face.geometry.surfaceType == adsk.core.SurfaceTypes.ConeSurfaceType:
            _mark_face(face, "side")
        elif face.geometry.origin.z == 0:
            _mark_face(face, "bottom")
        else:
            _mark_face(face, "top")


def box(x, y, z, *, name="Box"):
    x, y, z = _cm((x, y, z))
    brep = adsk.fusion.TemporaryBRepManager.get()
    box_body = brep.createBox(adsk.core.OrientedBoundingBox3D.create(
        adsk.core.Point3D.create(x/2, y/2, z/2),
        adsk.core.Vector3D.create(1, 0, 0),
        adsk.core.Vector3D.create(0, 1, 0),
        x, y, z))
    occurrence = _create_component(root(), box_body, name=name)

    def _find_and_mark_face(face_name, _x, _y, _z):
        face = occurrence.component.findBRepUsingPoint(
            adsk.core.Point3D.create(_x, _y, _z),
            adsk.fusion.BRepEntityTypes.BRepFaceEntityType)
        face = face.item(0)
        _mark_face(face, face_name)

    _find_and_mark_face("bottom", x/2, y/2, 0)
    _find_and_mark_face("top", x/2, y/2, z)
    _find_and_mark_face("left", 0, y/2, z/2)
    _find_and_mark_face("right", x, y/2, z/2)
    _find_and_mark_face("front", x/2, 0, z/2)
    _find_and_mark_face("back", x/2, y, z/2)

    return occurrence


def rect(x, y, *, name="Rectangle"):
    (x, y) = _cm((x, y))
    brep = adsk.fusion.TemporaryBRepManager.get()
    curves = [
        adsk.core.Line3D.create(
            adsk.core.Point3D.create(0, 0, 0),
            adsk.core.Point3D.create(x, 0, 0)
        ),
        adsk.core.Line3D.create(
            adsk.core.Point3D.create(x, 0, 0),
            adsk.core.Point3D.create(x, y, 0)
        ),
        adsk.core.Line3D.create(
            adsk.core.Point3D.create(x, y, 0),
            adsk.core.Point3D.create(0, y, 0)
        ),
        adsk.core.Line3D.create(
            adsk.core.Point3D.create(0, y, 0),
            adsk.core.Point3D.create(0, 0, 0)
        )
    ]
    wire, _ = brep.createWireFromCurves(curves)
    face = brep.createFaceFromPlanarWires([wire])
    return _create_component(root(), face, name=name)


def circle(r, *, name="Circle"):
    brep = adsk.fusion.TemporaryBRepManager.get()
    circle = adsk.core.Circle3D.createByCenter(
        adsk.core.Point3D.create(0, 0, 0),
        adsk.core.Vector3D.create(0, 0, 1),
        _cm(r)
    )
    wire, _ = brep.createWireFromCurves([circle])
    face = brep.createFaceFromPlanarWires([wire])
    return _create_component(root(), face, name=name)


def loft(*sketches):
    loft_input = root().features.loftFeatures.createInput(adsk.fusion.FeatureOperations.NewComponentFeatureOperation)
    for sketch in sketches:
        if sketch.profiles.count > 1:
            raise ValueError("Sketch %s contains multiple profiles" % sketch.name)
        loft_input.loftSections.add(sketch.profiles.item(0))
    feature = root().features.loftFeatures.add(loft_input)
    return root().allOccurrencesByComponent(feature.parentComponent)[0]


def _do_intersection(target_occurrence, tool_bodies):
    for target_body in _occurrence_bodies(target_occurrence):
        combine_input = target_occurrence.component.features.combineFeatures.createInput(target_body, tool_bodies)
        combine_input.operation = adsk.fusion.FeatureOperations.IntersectFeatureOperation
        combine_input.isKeepToolBodies = True
        target_occurrence.component.features.combineFeatures.add(combine_input)


def intersection(*occurrences, name=None):
    base_occurrence = occurrences[0]

    plane = None
    if _check_2D(base_occurrence):
        plane = _get_plane(base_occurrence)

    result_occurrence = _get_parent_component(base_occurrence).occurrences.addNewComponent(adsk.core.Matrix3D.create())
    result_occurrence.component.name = name or base_occurrence.component.name

    for body in _occurrence_bodies(base_occurrence):
        body.copyToComponent(result_occurrence)

    for tool_occurrence in occurrences[1:]:
        if _check_2D(tool_occurrence):
            plane = _check_coplanarity(plane, _get_plane(tool_occurrence))
        _do_intersection(result_occurrence, _collection_of(_occurrence_bodies(tool_occurrence)))

    for occurrence in occurrences:
        occurrence.moveToComponent(result_occurrence)
        occurrence = occurrence.createForAssemblyContext(result_occurrence)
        occurrence.isLightBulbOn = False
    if base_occurrence.assemblyContext is not None:
        result_occurrence = result_occurrence.createForAssemblyContext(base_occurrence.assemblyContext)
    return result_occurrence


def _do_difference(target_occurrence, tool_occurrence):
    tool_bodies = adsk.core.ObjectCollection.create()  # type: adsk.core.ObjectCollection
    for tool_body in _occurrence_bodies(tool_occurrence):
        tool_bodies.add(tool_body)

    for target_body in _occurrence_bodies(target_occurrence):
        combine_input = target_occurrence.component.features.combineFeatures.createInput(target_body, tool_bodies)
        combine_input.operation = adsk.fusion.FeatureOperations.CutFeatureOperation
        combine_input.isKeepToolBodies = True
        target_occurrence.component.features.combineFeatures.add(combine_input)


def difference(*occurrences, name=None):
    base_occurrence = occurrences[0]

    is2D = _check_2D(base_occurrence)
    plane = None

    result_occurrence = _get_parent_component(base_occurrence).occurrences.addNewComponent(adsk.core.Matrix3D.create())
    result_occurrence.component.name = name or base_occurrence.component.name
    for body in _occurrence_bodies(base_occurrence):
        if is2D:
            plane = _check_coplanarity(plane, _get_plane(body))
        body.copyToComponent(result_occurrence)

    try:
        for tool_occurrence in occurrences[1:]:
            if is2D:
                if _check_2D(tool_occurrence):
                    for body in _occurrence_bodies(tool_occurrence):
                        _check_coplanarity(plane, _get_plane(body))
            else:
                if _check_2D(tool_occurrence):
                    raise ValueError("Can't subtract 2D geometry from 3D geometry")
            _do_difference(result_occurrence, tool_occurrence)
    except ValueError:
        result_occurrence.deleteMe()
        raise

    for occurrence in occurrences:
        occurrence.moveToComponent(result_occurrence)
        occurrence = occurrence.createForAssemblyContext(result_occurrence)
        occurrence.isLightBulbOn = False
    if base_occurrence.assemblyContext is not None:
        result_occurrence = result_occurrence.createForAssemblyContext(base_occurrence.assemblyContext)
    return result_occurrence


def translate(occurrence, x=0, y=0, z=0):
    if x == 0 and y == 0 and z == 0:
        return occurrence

    bodies_to_move = adsk.core.ObjectCollection.create()
    for body in _occurrence_bodies(occurrence):
        bodies_to_move.add(body)

    transform = adsk.core.Matrix3D.create()
    transform.translation = _cm(adsk.core.Vector3D.create(x, y, z))

    original_transform = occurrence.transform  # type: adsk.core.Matrix3D
    original_transform.transformBy(transform)
    occurrence.transform = original_transform
    design().snapshots.add()
    return occurrence


def rotate(occurrence, x=0, y=0, z=0, center=None):
    if x == 0 and y == 0 and z == 0:
        return occurrence

    if center is None:
        center = adsk.core.Point3D.create(0, 0, 0)
    else:
        center = _cm(adsk.core.Point3D.create(*center))

    bodies_to_rotate = adsk.core.ObjectCollection.create()
    for body in _occurrence_bodies(occurrence):
        bodies_to_rotate.add(body)

    transform1 = adsk.core.Matrix3D.create()
    transform1.setToRotation(math.radians(x), adsk.core.Vector3D.create(1, 0, 0), center)
    transform2 = adsk.core.Matrix3D.create()
    transform2.setToRotation(math.radians(y), adsk.core.Vector3D.create(0, 1, 0), center)
    transform3 = adsk.core.Matrix3D.create()
    transform3.setToRotation(math.radians(z), adsk.core.Vector3D.create(0, 0, 1), center)

    transform1.transformBy(transform2)
    transform1.transformBy(transform3)

    transform = occurrence.transform  # type: adsk.core.Matrix3D
    transform.transformBy(transform1)
    occurrence.transform = transform
    design().snapshots.add()

    return occurrence


def component(*occurrences, name="Component") -> adsk.fusion.Occurrence:
    new_occurrence = root().occurrences.addNewComponent(adsk.core.Matrix3D.create())  # type: adsk.fusion.Occurrence
    new_component = new_occurrence.component  # type: adsk.fusion.Component
    new_component.name = name

    for occurrence in occurrences:
        occurrence.moveToComponent(new_occurrence)
    return new_occurrence


def _get_plane(entity):
    if isinstance(entity, adsk.fusion.BRepBody):
        body = entity
        if body.isSolid:
            raise ValueError("Can't get the plane of a 3D object")
        plane = None
        for face in body.faces:
            if not isinstance(face.geometry, adsk.core.Plane):
                raise ValueError("Can't get the plane of a non-planar face")
            if plane is None:
                plane = face.geometry
            else:
                _check_coplanarity(plane, face.geometry)
        return plane
    else:
        plane = None
        for body in _occurrence_bodies(entity):
            if plane is None:
                plane = _get_plane(body)
            else:
                _check_coplanarity(plane, _get_plane(body))
        return plane


def union(*occurrences, name=None):
    is2D = None
    plane = None
    bodies = []
    for occurrence in occurrences:
        if is2D is None:
            is2D = _check_2D(occurrence)
        elif is2D != _check_2D(occurrence):
            raise ValueError("Can't union 2D and 3D geometry")

        if is2D:
            plane = _check_coplanarity(plane, _get_plane(occurrence))

        bodies.extend(_occurrence_bodies(occurrence))

    base_occurrence = occurrences[0]

    parent_component = _get_parent_component(base_occurrence)
    result_occurrence = parent_component.occurrences.addNewComponent(adsk.core.Matrix3D.create())

    if len(bodies) > 1:
        body_copies = []
        for body in bodies:
            body_copies.append(body.copyToComponent(result_occurrence))

        combine_input = parent_component.features.combineFeatures.createInput(
            body_copies[0], _collection_of(body_copies[1:]))
        combine_input.operation = adsk.fusion.FeatureOperations.JoinFeatureOperation
        combine_input.isKeepToolBodies = False
        combine_input.isNewComponent = False
        parent_component.features.combineFeatures.add(combine_input)
        for occurrence in occurrences:
            occurrence.moveToComponent(result_occurrence)
            occurrence = occurrence.createForAssemblyContext(result_occurrence)
            occurrence.isLightBulbOn = False
    else:
        for occurrence in occurrences:
            occurrence.moveToComponent(result_occurrence)
    result_occurrence.component.name = name or base_occurrence.component.name

    if base_occurrence.assemblyContext is not None:
        result_occurrence = result_occurrence.createForAssemblyContext(base_occurrence.assemblyContext)
    return result_occurrence


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

    translate(target_occurrence, *_mm((
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


def duplicate(func, values, occurrence):
    result_occurrence = root().occurrences.addNewComponent(adsk.core.Matrix3D.create())
    result_occurrence.component.name = occurrence.name

    occurrence.moveToComponent(result_occurrence)
    occurrence = occurrence.createForAssemblyContext(result_occurrence)
    func(occurrence, values[0])
    for value in values[1:]:
        duplicate_occurrence = result_occurrence.component.occurrences.addExistingComponent(
            occurrence.component, adsk.core.Matrix3D.create())
        func(duplicate_occurrence, value)

    return result_occurrence


def place(occurrence, x_placement=keep(), y_placement=keep(), z_placement=keep()) -> adsk.fusion.Occurrence:
    bounding_box = _get_exact_bounding_box(occurrence)
    translate(occurrence,
              x_placement(0, bounding_box),
              y_placement(1, bounding_box),
              z_placement(2, bounding_box))
    return occurrence


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
        if key == "run" or key == "stop":
            continue
        fscad.__setattr__(key, value)


def stop(context):
    del sys.modules['fscad']
