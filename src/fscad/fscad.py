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

__all__ = ['app', 'root', 'ui', 'brep', 'design', 'Translation', 'Place', 'BoundedEntity', 'BRepEntity', 'Body', 'Loop',
           'Face', 'Edge', 'Point', 'BoundingBox', 'Component', 'ComponentWithChildren', 'Shape', 'BRepComponent',
           'PlanarShape', 'Box', 'Cylinder', 'Sphere', 'Torus', 'Rect', 'Circle', 'Builder2D', 'Polygon',
           'RegularPolygon', 'import_fusion_archive', 'import_dxf', 'Combination', 'Union', 'Difference',
           'Intersection', 'Group', 'Loft', 'Revolve', 'Sweep', 'ExtrudeBase', 'Extrude', 'ExtrudeTo', 'OffsetEdges',
           'SplitFace', 'Silhouette', 'Hull', 'RawThreads', 'Threads', 'Fillet', 'Chamfer', 'Scale', 'Thicken',
           'MemoizableDesign', 'setup_document', 'run_design', 'relative_import']

import functools
import importlib
import inspect
import math
import os
import pathlib
import random
import sys
import time
import traceback
import types
from abc import ABC
from typing import Callable, Iterable, Iterator, Mapping, Optional, Sequence, Tuple, List, TypeVar
from typing import Union as Onion  # Haha, why not? Prevents a conflict with our Union type

import adsk.core
import adsk.fusion
from adsk.core import BoundingBox3D, Curve3D, Line3D, Matrix3D, NurbsCurve3D, ObjectCollection, OrientedBoundingBox3D, \
    Point2D, Point3D, ValueInput, Vector3D
from adsk.fusion import BRepBody, BRepCoEdge, BRepEdge, BRepFace, BRepLoop, Occurrence, SketchCircle, SketchCurve, \
    SketchEllipse

# recursive type hints don't actually work yet, so let's expand the recursion a few levels and call it good
_singular_face_selector_types = Onion['Component', 'Body', 'Face']
_face_selector_types = Onion[_singular_face_selector_types, Iterable[
    Onion[_singular_face_selector_types, Iterable[
        Onion[_singular_face_selector_types, Iterable[
            Onion[_singular_face_selector_types, Iterable['_face_selector_types']]]]]]]]

# recursive type hints don't actually work yet, so let's expand the recursion a few levels and call it good
_singular_edge_selector_types = Onion['Component', 'Body', 'Face', 'Edge']
_edge_selector_types = Onion[_singular_edge_selector_types, Iterable[
    Onion[_singular_edge_selector_types, Iterable[
        Onion[_singular_edge_selector_types, Iterable[
            Onion[_singular_edge_selector_types, Iterable['_edge_selector_types']]]]]]]]

_singular_entity_types = Onion['Component', 'Body', 'Face', 'Edge', 'Point']
_entity_types = Onion[_singular_entity_types, Iterable[
    Onion[_singular_entity_types, Iterable[
        Onion[_singular_entity_types, Iterable[
            Onion[_singular_entity_types, Iterable['_entity_types']]]]]]]]

_vector_like = Onion[Vector3D, 'Translation', Point3D]

_brep_types = Onion[adsk.fusion.BRepBody, adsk.fusion.BRepCell, adsk.fusion.BRepCoEdge, adsk.fusion.BRepEdge,
                    adsk.fusion.BRepFace, adsk.fusion.BRepLoop, adsk.fusion.BRepLump, adsk.fusion.BRepShell,
                    adsk.fusion.BRepVertex, adsk.fusion.BRepWire]


def app():
    return adsk.core.Application.get()


def root() -> adsk.fusion.Component:
    return design().rootComponent


def ui():
    return app().userInterface


_brep = None


def brep():
    # caching the brep is a workaround for a weird bug where an exception from calling a TemporaryBRepManager method
    # and then catching the exception causes TemporaryBRepManager.get() to then throw the same error that was previously
    # thrown and caught. Probably some weird SWIG bug or something.
    global _brep
    if not _brep:
        _brep = adsk.fusion.TemporaryBRepManager.get()
    return _brep


def design() -> adsk.fusion.Design:
    return adsk.fusion.Design.cast(app().activeProduct)


def _collection_of(collection):
    object_collection = ObjectCollection.create()
    for obj in collection:
        object_collection.add(obj)
    return object_collection


def _create_component(parent_component, *bodies: Onion[BRepBody, 'Body'], name) -> Occurrence:
    new_occurrence = parent_component.occurrences.addNewComponent(Matrix3D.create())
    new_occurrence.component.name = name
    for body in bodies:
        if isinstance(body, Body):
            body = body.brep
        new_occurrence.component.bRepBodies.add(body)
    return new_occurrence


def _oriented_bounding_box_to_bounding_box(oriented: OrientedBoundingBox3D):
    coordinate_transform = Matrix3D.create()
    coordinate_transform.setToAlignCoordinateSystems(
        Point3D.create(0, 0, 0),
        oriented.lengthDirection,
        oriented.widthDirection,
        oriented.heightDirection,
        Point3D.create(0, 0, 0),
        Vector3D.create(1, 0, 0),
        Vector3D.create(0, 1, 0),
        Vector3D.create(0, 0, 1))
    center = oriented.centerPoint.copy()
    center.transformBy(coordinate_transform)

    return BoundingBox3D.create(
        Point3D.create(
            center.x - oriented.length / 2.0,
            center.y - oriented.width / 2.0,
            center.z - oriented.height / 2.0),
        Point3D.create(
            center.x + oriented.length / 2.0,
            center.y + oriented.width / 2.0,
            center.z + oriented.height / 2.0)
    )


def _get_exact_bounding_box(entity):
    vector1 = adsk.core.Vector3D.create(1.0, 0.0, 0.0)
    vector2 = adsk.core.Vector3D.create(0.0, 1.0, 0.0)

    if isinstance(entity, Component):
        entities = entity.bodies
        # noinspection PyTypeChecker
        return _get_exact_bounding_box(entities)

    if isinstance(entity, BRepEntity):
        return _get_exact_bounding_box(entity.brep)

    if hasattr(entity, "objectType"):
        if entity.objectType.startswith("adsk::fusion::BRep"):
            return _oriented_bounding_box_to_bounding_box(
                app().measureManager.getOrientedBoundingBox(entity, vector1, vector2))
        else:
            raise TypeError("Cannot get bounding box for type %s" % type(entity).__name__)

    try:
        iter(entity)
    except TypeError:
        raise TypeError("Cannot get bounding box for type %s" % type(entity).__name__)
    entities = entity

    bounding_box = None
    for entity in entities:
        entity_bounding_box = _get_exact_bounding_box(entity)
        if bounding_box is None:
            bounding_box = entity_bounding_box
        else:
            bounding_box.combine(entity_bounding_box)
    return bounding_box


def _body_index(body: Onion[BRepBody, 'Body'], bodies: Iterable[Onion[BRepBody, 'Body']]) -> Optional[int]:
    if isinstance(body, Body):
        return _body_index(body.brep, bodies)
    for i, candidate_body in enumerate(bodies):
        if isinstance(candidate_body, Body):
            candidate_body = candidate_body.brep
        if candidate_body == body:
            return i
    return None


def _face_index(face: Onion[BRepFace, 'Face']) -> int:
    if isinstance(face, Face):
        return _face_index(face.brep)
    for i, candidate_face in enumerate(face.body.faces):
        if candidate_face == face:
            return i
    assert False


def _edge_index(edge: Onion[BRepEdge, 'Edge']):
    if isinstance(edge, Edge):
        return _edge_index(edge.brep)
    for i, candidate_edge in enumerate(edge.body.edges):
        if candidate_edge == edge:
            return i
    assert False


def _loop_index(loop: BRepLoop):
    for i, candidate_loop in enumerate(loop.face.loops):
        if candidate_loop == loop:
            return i
    assert False


def _map_face(face, new_body):
    return new_body.faces[_face_index(face)]


def _flatten_face_selectors(selector: _face_selector_types) -> Iterable[BRepFace]:
    faces = []
    if isinstance(selector, Iterable):
        for sub_selector in selector:
            for face in _flatten_face_selectors(sub_selector):
                if face not in faces:
                    faces.append(face)
        return faces
    if isinstance(selector, Component):
        return _flatten_face_selectors(selector.bodies)
    if isinstance(selector, Body):
        return selector.brep.faces
    return selector.brep,


def _flatten_edge_selectors(selector: _edge_selector_types) -> Iterable[Onion[BRepFace, BRepEdge]]:
    if isinstance(selector, Iterable):
        selectors = []
        for sub_selector in selector:
            for selector in _flatten_edge_selectors(sub_selector):
                if selector not in selectors:
                    selectors.append(selector)
        return selectors
    if isinstance(selector, Component):
        return _flatten_edge_selectors(selector.bodies)
    if isinstance(selector, Body):
        return selector.brep.faces
    if isinstance(selector, Face) or isinstance(selector, Edge):
        return selector.brep,
    if isinstance(selector, BRepFace) or isinstance(selector, BRepEdge):
        return selector,
    raise ValueError("Invalid selector type: %s" % type(selector))


def _union_entities(entity: _entity_types, result_body: BRepBody = None, vector: Vector3D = None) -> BRepBody:
    if isinstance(entity, Iterable):
        for sub_entity in entity:
            result_body = _union_entities(sub_entity, result_body)
        return result_body
    if isinstance(entity, Component):
        for body in entity.bodies:
            if result_body is None:
                result_body = brep().copy(body.brep)
            else:
                brep().booleanOperation(result_body, body.brep, adsk.fusion.BooleanTypes.UnionBooleanType)
        return result_body
    if isinstance(entity, BRepEntity):
        if result_body is None:
            return brep().copy(entity.brep)
        else:
            brep().booleanOperation(result_body, brep().copy(entity.brep), adsk.fusion.BooleanTypes.UnionBooleanType)
            return result_body
    if isinstance(entity, Point):
        body = _create_point_body(entity.point, vector)
        if result_body is None:
            return body
        else:
            brep().booleanOperation(result_body, body, adsk.fusion.BooleanTypes.UnionBooleanType)
            return result_body


def _check_face_intersection(entity1, entity2):
    entity1_copy = brep().copy(entity1)
    entity2_copy = brep().copy(entity2)
    brep().booleanOperation(entity1_copy, entity2_copy, adsk.fusion.BooleanTypes.IntersectionBooleanType)
    return entity1_copy.faces.count > 0


def _point_vector_to_line(point, vector):
    return adsk.core.Line3D.create(point,
                                   adsk.core.Point3D.create(point.x + vector.x,
                                                            point.y + vector.y,
                                                            point.z + vector.z))


def _get_arbitrary_perpendicular_unit_vector(vector: Vector3D):
    while True:
        random_vector = Vector3D.create(random.random(), random.random(), random.random())
        if not random_vector.isParallelTo(vector):
            break
    arbitrary_vector = random_vector.crossProduct(vector)
    arbitrary_vector.normalize()
    return arbitrary_vector


def _check_face_geometry(face1, face2):
    """Does some quick sanity checks of the face geometry, to rule out easy cases of non-equality.

    A return value of True does not guarantee the geometry is the same, but a return value of False does
    guarantee they are not.
    """
    geometry1 = face1.geometry
    geometry2 = face2.geometry
    if isinstance(geometry1, adsk.core.Cylinder):
        if not math.isclose(geometry1.radius, geometry2.radius):
            return False
        line1 = _point_vector_to_line(geometry1.origin, geometry1.axis)
        line2 = _point_vector_to_line(geometry2.origin, geometry2.axis)
        return line1.isColinearTo(line2)
    if isinstance(geometry1, adsk.core.Sphere):
        if not math.isclose(geometry1.radius, geometry2.radius):
            return False
        return geometry1.origin.isEqualTo(geometry2.origin)
    if isinstance(geometry1, adsk.core.Torus):
        if not geometry1.origin.isEqualTo(geometry2.origin):
            return False
        if not geometry1.axis.isParallelTo(geometry2.axis):
            return False
        if not math.isclose(geometry1.majorRadius, geometry2.majorRadius):
            return False
        return math.isclose(geometry1.minorRadius, geometry2.minorRadius)
    if isinstance(geometry1, adsk.core.EllipticalCylinder):
        line1 = _point_vector_to_line(geometry1.origin, geometry1.axis)
        line2 = _point_vector_to_line(geometry2.origin, geometry2.axis)
        if not line1.isColinearTo(line2):
            return False
        if not geometry1.majorAxis.isParallelTo(geometry2.majorAxis):
            return False
        if not math.isclose(geometry1.majorRadius, geometry2.majorRadius):
            return False
        return math.isclose(geometry1.minorRadius, geometry2.minorRadius)
    # It's a bit harder to check the remaining types. We'll just fallback to doing the
    # full face intersection check.
    return True


def _check_face_coincidence(face1, face2):
    if face1.geometry.surfaceType != face2.geometry.surfaceType:
        return False
    if face1.geometry.surfaceType == adsk.core.SurfaceTypes.PlaneSurfaceType:
        if not face1.geometry.isCoPlanarTo(face2.geometry):
            return False
        return _check_face_intersection(face1, face2)
    else:
        if not _check_face_geometry(face1, face2):
            return False
        return _check_face_intersection(face1, face2)


def _find_coincident_faces_on_body(body: BRepBody, selectors: Iterable[BRepFace]) -> Iterable[BRepFace]:
    coincident_faces = []
    candidate_selectors = []
    for selector in selectors:
        selector_bounding_box = selector.boundingBox
        expanded_bounding_box = adsk.core.BoundingBox3D.create(
            adsk.core.Point3D.create(
                selector_bounding_box.minPoint.x - app().pointTolerance,
                selector_bounding_box.minPoint.y - app().pointTolerance,
                selector_bounding_box.minPoint.z - app().pointTolerance),
            adsk.core.Point3D.create(
                selector_bounding_box.maxPoint.x + app().pointTolerance,
                selector_bounding_box.maxPoint.y + app().pointTolerance,
                selector_bounding_box.maxPoint.z + app().pointTolerance),
        )
        if body.boundingBox.intersects(expanded_bounding_box):
            candidate_selectors.append((selector, expanded_bounding_box))

    for body_face in body.faces:
        for selector, expanded_bounding_box in candidate_selectors:
            if body_face.boundingBox.intersects(expanded_bounding_box):
                if _check_face_coincidence(selector, body_face):
                    coincident_faces.append(body_face)
                    break
    return coincident_faces


def _check_edge_coincidence(entity1, entity2):
    entity1_copy = brep().copy(entity1)
    entity2_copy = brep().copy(entity2)
    brep().booleanOperation(entity1_copy, entity2_copy, adsk.fusion.BooleanTypes.IntersectionBooleanType)
    return entity1_copy.edges.count > 0


def _find_coincident_edges_on_body(
        body: BRepBody, selectors: Iterable[Onion[BRepFace, BRepEdge]]) -> Iterable[BRepEdge]:
    coincident_edges = []
    candidate_selectors = []
    for selector in selectors:
        selector_bounding_box = selector.boundingBox
        expanded_bounding_box = adsk.core.BoundingBox3D.create(
            adsk.core.Point3D.create(
                selector_bounding_box.minPoint.x - app().pointTolerance,
                selector_bounding_box.minPoint.y - app().pointTolerance,
                selector_bounding_box.minPoint.z - app().pointTolerance),
            adsk.core.Point3D.create(
                selector_bounding_box.maxPoint.x + app().pointTolerance,
                selector_bounding_box.maxPoint.y + app().pointTolerance,
                selector_bounding_box.maxPoint.z + app().pointTolerance),
        )
        if body.boundingBox.intersects(expanded_bounding_box):
            candidate_selectors.append((selector, expanded_bounding_box))

    for body_edge in body.edges:
        for selector, expanded_bounding_box in candidate_selectors:
            if body_edge.boundingBox.intersects(expanded_bounding_box):
                if _check_edge_coincidence(body_edge, selector):
                    coincident_edges.append(body_edge)
                    break
    return coincident_edges


def _point_3d(point: Onion[Point2D, Point3D, Tuple[float, float], Tuple[float, float, float], 'Point']):
    if isinstance(point, Point3D):
        return point
    if isinstance(point, Point2D):
        return Point3D.create(point.x, point.y, 0)
    if isinstance(point, Point):
        return point.point
    if isinstance(point, tuple):
        if len(point) >= 3:
            return Point3D.create(*point[0:3])
        elif len(point) == 2:
            return Point3D.create(*point, 0)
        else:
            raise ValueError("tuples must have at least 2 values")
    raise ValueError("Unsupported type: %s" % point.__class__.__name__)


def _point_2d(point: Onion[Point2D, Point3D, Tuple[float, float], Tuple[float, float, float], 'Point']) -> Point3D:
    point = _point_3d(point)
    if point.z != 0:
        raise ValueError("Only points in the x/y plane (z=0) are supported")
    return point


def _project_point_to_line(point: Point3D, line: adsk.core.InfiniteLine3D):
    axis = line.direction
    axis.normalize()

    point_to_line_origin = point.vectorTo(line.origin)

    # project the vector onto the line, and reverse it
    axis_projection = axis.copy()
    axis_projection.scaleBy(-1 * point_to_line_origin.dotProduct(axis))

    projected_point = line.origin
    projected_point.translateBy(axis_projection)
    return projected_point


def _create_empty_body() -> BRepBody:
    body1 = brep().createSphere(Point3D.create(0, 0, 0), 1.0)
    body2 = brep().createSphere(Point3D.create(10, 0, 0), 1.0)
    brep().booleanOperation(body1, body2, adsk.fusion.BooleanTypes.IntersectionBooleanType)
    return body1


def _create_point_body(point: Point3D, vector: Vector3D = None):
    if vector is None:
        vector = Vector3D.create(0, 0, app().pointTolerance * 2)
    else:
        vector = vector.copy()
        vector.normalize()
        vector.scaleBy(app().pointTolerance * 2)

    second_point = point.copy()
    second_point.translateBy(vector)

    body, _ = brep().createWireFromCurves([adsk.core.Line3D.create(point, second_point)])

    matrix = adsk.core.Matrix3D.create()
    translation = Matrix3D.create()
    translation.translation = point.asVector()
    translation.invert()
    matrix.transformBy(translation)

    # we can't directly create a wire with points closer than app().pointTolerance, but we can scale it afterward.
    # This should result in two points being .01 * app().pointTolerance within each other
    scale = Matrix3D.create()
    scale.setCell(0, 0, .005)
    scale.setCell(1, 1, .005)
    scale.setCell(2, 2, .005)
    matrix.transformBy(scale)

    translation.invert()
    matrix.transformBy(translation)

    brep().transform(body, matrix)

    return body


def _create_wire(curve: Curve3D) -> BRepBody:
    return brep().createWireFromCurves([curve], allowSelfIntersections=False)[0]


def _create_fit_point_spline(*points: Sequence[Onion[Tuple[float, float], Point2D, Point3D, 'Point']]) -> NurbsCurve3D:
    temp_occurrence = _create_component(root(), name="temp")

    construction_plane_input = temp_occurrence.component.constructionPlanes.createInput(temp_occurrence)
    construction_plane_input.setByPlane(adsk.core.Plane.create(
        Point3D.create(0, 0, 0),
        Vector3D.create(0, 0, 1)))
    construction_plane = temp_occurrence.component.constructionPlanes.add(construction_plane_input)
    sketch = temp_occurrence.component.sketches.add(construction_plane, temp_occurrence)

    spline = sketch.sketchCurves.sketchFittedSplines.add(_collection_of([_point_2d(point) for point in points]))
    curve = spline.worldGeometry

    temp_occurrence.deleteMe()
    return curve


def _get_outer_loop(brep_face: BRepFace) -> BRepLoop:
    loop: adsk.fusion.BRepLoop
    for loop in brep_face.loops:
        if loop.isOuter:
            return loop


def _find_connected_edge_endpoints(edges: Sequence['Edge'], face: 'Face') -> \
        Tuple[Optional[BRepCoEdge], Optional[BRepCoEdge], adsk.fusion.BRepLoop]:
    edges_to_process = list(edges)

    first_edge = edges_to_process.pop()
    start_coedge: adsk.fusion.BRepCoEdge = None
    for coedge in first_edge.brep.coEdges:
        if coedge.loop.face == face.brep:
            start_coedge = coedge
            break

    if not start_coedge:
        raise ValueError("The provided edges are not all on the provided face")

    end_coedge = start_coedge
    full_loop = False

    def remove_edge_to_process(edge_to_remove: BRepEdge):
        index_to_remove = None
        for i, edge_to_process in enumerate(edges_to_process):
            if edge_to_process.brep == edge_to_remove:
                index_to_remove = i
                break
        if index_to_remove is not None:
            del(edges_to_process[index_to_remove])
            return True
        return False

    while True:
        next_coedge = end_coedge.next

        if next_coedge == start_coedge:
            full_loop = True
            break

        if remove_edge_to_process(next_coedge.edge):
            end_coedge = next_coedge
        else:
            break

    if not full_loop:
        while True:
            prev_coedge = start_coedge.previous

            if remove_edge_to_process(prev_coedge.edge):
                start_coedge = prev_coedge
            else:
                break

    if edges_to_process:
        raise ValueError("All edges must be on the same face and connected")
    if full_loop:
        return None, None, start_coedge.loop
    return start_coedge, end_coedge, start_coedge.loop


def _find_coedge_for_face(edge: BRepEdge, face: BRepFace) -> BRepCoEdge:
    for coedge in edge.coEdges:
        if coedge.loop.face == face:
            return coedge
    raise ValueError("The face is not associated with the given edge")


def _create_construction_point(component: adsk.fusion.Component, point: Point3D):
    input = component.constructionPoints.createInput()
    input.setByPoint(point)
    return component.constructionPoints.add(input)


def _pairwise_indices(count):
    for i in range(0, count):
        yield i, (i+1) % count


def _iterate_pairwise(collection):
    for i, j in _pairwise_indices(len(collection)):
        yield collection[i], collection[j]


def _add_face_def(shell_def: adsk.fusion.BRepShellDefinition, surface_geometry, edges):
    face_def = shell_def.faceDefinitions.add(surface_geometry, isParamReversed=False)
    loop_def = face_def.loopDefinitions.add()
    for edge in edges:
        loop_def.bRepCoEdgeDefinitions.add(edge, False)


class Translation(object):
    """A wrapper around a Vector3D, that provides functionality useful for use with the `Component.place` method.

    This is primarily used as an intermediate object within a placement expression within a Component.place() call. It
    allows specifying an additive or multiplicative offset to the base placement vector. This object is typically
    returned by the Place object's '==' operator overload.

    Args:
        vector: The Vector3D that this Translation object wraps.
    """
    def __init__(self, vector: Vector3D):
        self._vector = vector

    def vector(self) -> Vector3D:
        """Return: The Vector3D that this Translation wraps. """
        return self.vector

    def __add__(self, other: float) -> 'Translation':
        """Modifies this Translation object by adding the given constant to every component.

        Returns: This Translation object with the specified addition applied."""
        self._vector.setWithArray((self._vector.x + other, self._vector.y + other, self._vector.z + other))
        return self

    def __sub__(self, other: float) -> 'Translation':
        """Modifies this Translation object by subtracting the given value from every component.

        Returns: This Translation object with the specified subtraction applied."""
        self._vector.setWithArray((self._vector.x - other, self._vector.y - other, self._vector.z - other))
        return self

    def __mul__(self, other: float) -> 'Translation':
        """Modifies this Translation object by multiplying every component by the given value.

        Returns: This Translation object with the specified multiplication applied."""
        self._vector.setWithArray((self._vector.x * other, self._vector.y * other, self._vector.z * other))
        return self

    def __div__(self, other: float) -> 'Translation':
        """Modifies this Translation object by dividing every component by the given value.

        Returns: This Translation object with the specified division applied."""
        self._vector.setWithArray((self._vector.x / other, self._vector.y / other, self._vector.z / other))
        return self

    @property
    def x(self) -> float:
        """Returns: the X component of this Translation."""
        return self._vector.x

    @property
    def y(self) -> float:
        """Returns: the Y component of this Translation."""
        return self._vector.y

    @property
    def z(self) -> float:
        """Returns: the Z component of this Translation."""
        return self._vector.z


class Place(object):
    """An intermediate object intended for use with the `Component.place()` method.

    The idea is that you can get the Place object for various defined points on an object, and then use the `==`
    operator with another Place or point-like object to get a vector between them, which can be used to "place" one
    object as some location relative to another.

    You typically get a Place object via the '-', '+' and '~' unary operator overloads on a BoundedEntity object.

    Args:
        point: The Point3D that this Place object represents.
    """
    def __init__(self, point: Point3D):
        self._point = point

    def __eq__(self, other: Onion['Place', float, int, Point3D]) -> Translation:
        if isinstance(other, Point3D):
            point = other
        elif isinstance(other, float) or isinstance(other, int):
            point = Point3D.create(other, other, other)
        elif isinstance(other, Place):
            point = other._point
        elif isinstance(other, Point):
            point = other.point
        else:
            raise ValueError("Unsupported type: %s" % type(other).__name__)

        return Translation(self._point.vectorTo(point))


class BoundedEntity(object):
    """This is the general superclass for any geometry object.

    This provides some common functionality for all geometry objects, like the size, bounding box, etc.

    It also provides the '-', '+' and '~' operator overloads that are intended for use by the `Component.place()`
    method
    """
    def __init__(self):
        self._cached_bounding_box = None
        self._cache_key = None

    @property
    def bounding_box(self) -> 'BoundingBox':
        """Returns: The bounding box of this entity."""
        cache_key = self._get_cache_key()
        if self._cache_key != cache_key:
            self._reset_cache()
        if self._cached_bounding_box is None:
            self._cached_bounding_box = BoundingBox(self._calculate_bounding_box())
            self._cache_key = cache_key
        return self._cached_bounding_box

    def _get_cache_key(self):
        return 0

    def _calculate_bounding_box(self) -> 'BoundingBox3D':
        raise NotImplementedError()

    def _reset_cache(self):
        self._cached_bounding_box = None

    def size(self) -> Vector3D:
        """Returns: The size of this entity as a Vector3D."""
        return self.bounding_box.raw_bounding_box.minPoint.vectorTo(
            self.bounding_box.raw_bounding_box.maxPoint)

    def min(self) -> Point3D:
        """Returns: The minimum point of this entity's bounding box."""
        return self.bounding_box.raw_bounding_box.minPoint

    def max(self) -> Point3D:
        """Returns: The maximum point of this entity's bounding box."""
        return self.bounding_box.raw_bounding_box.maxPoint

    def mid(self) -> Point3D:
        """Returns: The geometric midpoint of this entity."""
        return Point3D.create(
            (self.bounding_box.raw_bounding_box.minPoint.x + self.bounding_box.raw_bounding_box.maxPoint.x)/2,
            (self.bounding_box.raw_bounding_box.minPoint.y + self.bounding_box.raw_bounding_box.maxPoint.y)/2,
            (self.bounding_box.raw_bounding_box.minPoint.z + self.bounding_box.raw_bounding_box.maxPoint.z)/2)

    def __neg__(self) -> Place:
        """Returns: a Place object that represents this entity's negative bound."""
        return Place(self.min())

    def __pos__(self) -> Place:
        """Returns: a Place object that represents this entity's positive bound."""
        return Place(self.max())

    def __invert__(self) -> Place:
        """Returns: a Place object that represents this entity's midpoint."""
        return Place(self.mid())


class BRepEntity(BoundedEntity, ABC):
    """Represents a single BRep object.

    This is the superclass of the various wrappers around the raw Fusion 360 Brep* related objects.
    """
    def __init__(self, component: 'Component'):
        super().__init__()
        self._component = component

    @property
    def brep(self) -> _brep_types:
        """Returns: the raw BRep type that this object wraps"""
        raise NotImplementedError

    @property
    def component(self) -> 'Component':
        """Returns: The Component that this entity is a part of"""
        return self._component

    def _calculate_bounding_box(self) -> 'BoundingBox3D':
        return _get_exact_bounding_box(self.brep)

    def _get_cache_key(self):
        return self.brep

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.brep == other.brep

    def __ne__(self, other):
        if not isinstance(other, self.__class__):
            return True
        return self.brep != other.brep


T = TypeVar('T')


class _VirtualSequence(Sequence[T]):
    def __init__(self, indices: Sequence[int], producer: Callable[[int], T]):
        self._indices = indices
        self._producer = producer

    def __getitem__(self, i: Onion[int, slice]) -> Onion['Edge', Sequence['Edge']]:
        if isinstance(i, slice):
            return _VirtualSequence(self._indices[i], self._producer)
        return self._producer(i)

    def __iter__(self) -> Iterator['Edge']:
        return _SequenceIterator(self)

    def __len__(self) -> int:
        return len(self._indices)


class Body(BRepEntity):
    """Represents a single Body.

    This is a wrapper around Fusion 360's BRepBody object.

    Args:
        brep_provider: A Callable that returns the BRepBody for this body
        component: The Component that this BRepBody is a part of
    """
    def __init__(self, brep_provider: Callable[[], BRepBody], component: 'Component'):
        super().__init__(component)
        self._brep_provider = brep_provider

    @property
    def brep(self) -> BRepBody:
        """Returns: The raw BRepBody this object wraps."""
        return self._brep_provider()

    @property
    def edges(self) -> Sequence['Edge']:
        """Returns: All Edges that are a part of this Body, or an empty Sequence if there are None."""
        return _VirtualSequence(range(0, len(self.brep.edges)), lambda i: Edge(lambda: self.brep.edges[i], self))

    @property
    def faces(self) -> Sequence['Face']:
        """Returns: All Faces that are a part of this Body, or an empty Sequence if there are None."""
        return _VirtualSequence(range(0, len(self.brep.faces)), lambda i: Face(lambda: self.brep.faces[i], self))


class Loop(BRepEntity):
    """Represents a single Loop.

    This is a wrapper around Fusion 360's BRepLoop object.

    Args:
          brep_provider: A Callable that returns the BRepLoop for this loop
          face: The Face object that the BRepLoop is a part of
    """
    def __init__(self, brep_provider: Callable[[], BRepLoop], face: 'Face'):
        super().__init__(face.component)
        self._brep_provider = brep_provider
        self._face = face

    @property
    def brep(self) -> BRepLoop:
        """Returns: The raw BRepLoop this object wraps."""
        return self._brep_provider()

    @property
    def face(self) -> 'Face':
        """Returns: The Face object that this loop is a part of."""
        return self._face

    @property
    def edges(self) -> Sequence['Edge']:
        """Returns: All Edges that make up this loop."""
        return _VirtualSequence(
            range(0, len(self.brep.edges)),
            lambda i: Edge(lambda: self.brep.edges[i], self.face.body))

    @property
    def is_outer(self) -> bool:
        return self.brep.isOuter

    def get_plane(self) -> Optional[adsk.core.Plane]:
        """Returns: The plane that this Loop lies in, or None if this is not a planar component."""
        return self.face.get_plane()


class Face(BRepEntity):
    """Represents a single Face.

    This is a wrapper around Fusion 360's BRepFace object.

    Args:
        brep_provider: A Callable that returns the BRepFace for this face
        body: The Body object that the BRepFace is a part of
    """
    def __init__(self, brep_provider: Callable[[], BRepFace], body: Body):
        super().__init__(body.component)
        self._brep_provider = brep_provider
        self._body = body

    @property
    def brep(self) -> BRepFace:
        """Returns: The raw BRepFace this object wraps."""
        return self._brep_provider()

    @property
    def body(self) -> Body:
        """Returns: The Body object that this face is a part of."""
        return self._body

    @property
    def connected_faces(self) -> Sequence['Face']:
        """Returns: All Faces that are connected to this Face, or an empty Sequence if there are None."""
        result = []
        for edge in self.brep.edges:
            for face in edge.faces:
                if face != self.brep:
                    if face not in result:
                        result.append(face)
                    break
        result_faces = []

        for face in result:
            face_index = _face_index(face)
            result_faces.append(Face(lambda index=face_index: self._body.brep.faces[index], self._body))
        return result_faces

    @property
    def edges(self) -> Sequence['Edge']:
        """Returns: All Edges that make up the boundary of this Face, or an empty Sequence if there are None."""
        return _VirtualSequence(range(0, len(self.brep.edges)), lambda i: Edge(lambda: self.brep.edges[i], self.body))

    @property
    def outer_edges(self) -> Sequence['Edge']:
        """Returns: All Edges that make up the outer boundary of this Face, or an empty Sequence if there are none."""
        loop = _get_outer_loop(self.brep)
        loop_index = _loop_index(loop)
        return _VirtualSequence(
            range(0, len(loop.edges)),
            lambda i: Edge(lambda: self.brep.loops[loop_index].edges[i], self.body))

    @property
    def loops(self) -> Sequence[Loop]:
        """Returns: All Loops in this Face, or an empty Sequence if there are none."""
        return _VirtualSequence(
            range(0, len(self.brep.loops)),
            lambda i: Loop(lambda: self.brep.loops[i], self))

    def get_plane(self) -> Optional[adsk.core.Plane]:
        """Returns: The plane that this Face lies in, or None if this is not a planar component."""
        if isinstance(self.brep.geometry, adsk.core.Plane):
            if self.brep.isParamReversed:
                normal = self.brep.geometry.normal.copy()
                normal.scaleBy(-1)
                return adsk.core.Plane.create(
                    self.brep.geometry.origin,
                    normal)
            else:
                return self.brep.geometry
        return None

    def make_component(self, name="Face") -> 'Component':
        """Create a separate component that consists of just this face.

        Args:
              name: (optional) The name of the new component.
        Returns:
              A new component that consists of just this face. It will also contain the component containing this face
              as a non-visible child."""
        return BRepComponent(self.brep, component=self.component, name=name)


class Edge(BRepEntity):
    """Represents a single Edge.

    This is a wrapper around Fusion 360's BRepEdge object.

    Args:
        brep_provider: A Callable that returns the BRepEdge for this edge
        body: The Body object that the BRepEdge is a part of
    """
    def __init__(self, brep_provider: Callable[[], BRepEdge], body: Body):
        super().__init__(body.component)
        self._brep_provider = brep_provider
        self._body = body

    @property
    def brep(self) -> BRepEdge:
        """Returns: The raw BRepEdge this object wraps."""
        return self._brep_provider()

    @property
    def body(self) -> Body:
        """Returns: The Body object that this edge is a part of."""
        return self._body

    @property
    def faces(self) -> Sequence[Face]:
        """Returns: The faces associated with this edge."""
        return _VirtualSequence(range(0, len(self.brep.faces)),
                                lambda i: self.body.faces[_face_index(self.brep.faces[i])])


class Point(BoundedEntity):
    """Represents a single 3D Point.

    This class is a wrapper around Fusion 360's Point3D class.

    This is useful, for example, for being able to align a Component relative to a Point via the Component.place()
    method. e.g. component.place(~component == point, +component == point, -component == point)

    Args:
        point: The Point3D to wrap."""
    def __init__(self, point: Point3D):
        super().__init__()
        self._point = point

    @property
    def point(self) -> Point3D:
        """Returns the raw Point3D object that this class wraps."""
        return self._point.copy()

    @property
    def x(self) -> float:
        return self._point.x

    @property
    def y(self) -> float:
        return self._point.y

    @property
    def z(self) -> float:
        return self._point.z

    def _calculate_bounding_box(self) -> 'BoundingBox3D':
        return BoundingBox3D.create(self._point, self._point)


class BoundingBox(BoundedEntity):
    """Represents a bounding box of another entity or set of entities.

    This class is a wrapper around Fusion 360's BoundingBox3D object.

    Args:
        bounding_box: The BoundingBox3D to wrap.
    """
    def __init__(self, bounding_box: BoundingBox3D):
        super().__init__()
        self._bounding_box = bounding_box

    def _calculate_bounding_box(self) -> BoundingBox3D:
        return self._bounding_box

    @property
    def bounding_box(self) -> 'BoundingBox':
        return self

    @property
    def raw_bounding_box(self) -> 'BoundingBox3D':
        """Returns: a copy of the raw BoundingBox3D that this object wraps."""
        return self._bounding_box.copy()

    def make_box(self, name=None) -> 'Box':
        """Makes a Box component the same size and in the same location of this BoundingBox.

        Returns: The new Box component.
        """

        if self.size().x == 0:
            box = Rect(self.size().z, self.size().y, name=name)
            box.ry(90)
        elif self.size().y == 0:
            box = Rect(self.size().x, self.size().z, name=name)
            box.rx(90)
        elif self.size().z == 0:
            box = Rect(self.size().x, self.size().y, name=name)
        else:
            box = Box(*self.size().asArray(), name=name)

        box.place(
            ~box == ~self,
            ~box == ~self,
            ~box == ~self)
        return box

    def __str__(self) -> str:
        return "(%s, %s)" % (self._bounding_box.minPoint.asArray(), self._bounding_box.maxPoint.asArray())


class Component(BoundedEntity, ABC):
    """The top level Object of an fscad design.

    A component is made up of one or more Bodies, and may also contain visible and hidden children. When constructing
    one of the Component subclasses that are based an an input Component, that input Component will typically be made
    a hidden child of the newly created Component.

    The only case of a visible child currently is in the Group Component.

    This roughly corresponds with a Component or Occurrence in Fusion's API/UI. Except that there is no corresponding
    concept of having multiple Occurrences of a single Component. Instead, if a Component is used in multiple place in
    the design, multiple copies of the Component must be made.

    Creating a Component in itself doesn't actually create any object in the Fusion 360 document. You must call the
    create_occurrence method to actually create the component in the document. This is typically the last operation
    you would perform on the final top-level design component, after building it up from multiple sub-Components.
    It can also be useful to visualize any intermediate Components for debugging purposes.

    The reason for this is directly creating an object in the Fusion 360 document for every component/operation tends
    to get very slow for even slightly complex objects. Instead, fscad tries to use temporary Brep objects as much as
    possible, which are typically much faster to work with.
    """
    _origin = Point3D.create(0, 0, 0)
    _null_vector = Vector3D.create(0, 0, 0)
    _pos_x = Vector3D.create(1, 0, 0)
    _pos_y = Vector3D.create(0, 1, 0)
    _pos_z = Vector3D.create(0, 0, 1)

    """The name of the Component.
    
    When this Component is created as an Occurrence in the Fusion 360 Document, this will be used as the name of the
    Occurrence.
    """
    name = ...  # type: Optional[str]

    def __init__(self, name: str = None):
        super().__init__()
        self._parent = None
        self._local_transform = Matrix3D.create()
        self.name = name
        self._cached_brep_bodies = None
        self._cached_world_transform = None
        self._cached_inverse_transform = None
        self._named_points = {}
        self._named_edges = {}
        self._named_faces = {}

    def _calculate_bounding_box(self) -> BoundingBox3D:
        return _get_exact_bounding_box(self.bodies)

    def _raw_bodies(self) -> Iterable[BRepBody]:
        raise NotImplementedError()

    def copy(self, copy_children=True, name=None) -> 'Component':
        """Makes a copy of this Component.

        Args:
            copy_children: If true, the entire Component hierarchy under this Component is also copied. If false,
                only "visible" children are copied. Group is currently the only Component type that has visible
                children.
            name: If specified, set the name of the copy to name

        Returns: A new copy of this Component.
        """
        copy = Component()
        copy.__class__ = self.__class__
        copy._local_transform = self.world_transform()
        copy._cached_bounding_box = None
        copy._cached_brep_bodies = None
        copy._cached_world_transform = None
        copy._cached_inverse_transform = None
        copy._named_points = dict(self._named_points)
        copy._named_edges = dict(self._named_edges)
        copy._named_faces = dict(self._named_faces)
        copy.name = name or self.name
        self._copy_to(copy, copy_children)
        return copy

    def _copy_to(self, copy: 'Component', copy_children: bool):
        raise NotImplementedError

    def children(self) -> Iterable['Component']:
        """Returns: All direct children of this Component."""
        return ()

    def find_children(self, name, recursive=True) -> Sequence['Component']:
        """Find any children of this Component with the given name.

        Args:
            name: The name of the children to find
            recursive: If true, look for all children anywhere in the Component hierarchy under this Component. If
                false, only look for the direct children of this Component.

        Returns: The children with the given name, or an empty Sequence if none are found.
        """
        return ()

    def _default_name(self) -> str:
        return self.__class__.__name__

    @property
    def parent(self) -> Optional['Component']:
        """Returns: The parent Component of this Component, or None if this Component is a top-level Component with no
            parent.
        """
        return self._parent

    def _get_cached_brep_bodies(self):
        if self._cached_brep_bodies is None:
            world_transform = self.world_transform()
            raw_bodies_copy = [brep().copy(body) for body in self._raw_bodies()]
            for raw_body in raw_bodies_copy:
                brep().transform(raw_body, world_transform)
            self._cached_brep_bodies = tuple(raw_bodies_copy)
        return self._cached_brep_bodies

    @property
    def bodies(self) -> Sequence[Body]:
        """Returns: All bodies that make up this Component."""
        return _VirtualSequence(
            range(0, len(self._get_cached_brep_bodies())),
            lambda i: Body(lambda: self._get_cached_brep_bodies()[i], self))

    def create_occurrence(self, create_children=False, scale=1) -> adsk.fusion.Occurrence:
        """Creates an occurrence of this Component in the root of the document in Fusion 360.

        No objects are actually added to the Fusion 360 design until this method is called. This should be called
        once for every top level Component in your design.

        It is typically much quicker to create a component without children, so this is typically what you would want to
        do, unless there is a specific reason you need the children.

        Creating the children can be useful for debugging purposes, to be able to drill down into the design to find a
        specific sub-component in the context of the overall design.

        Args:
            create_children: If true, also add any children of this Component recursively as hidden children of the
                corresponding Occurrence in the Fusion 360 document.
            scale: If specified, the Occurrence in Fusion 360 will be created at the given scale, with the scale
                operation centered at the origin. This is most useful if you want to normally work in mm units, since
                Fusion 360's default unit for the API is cm. You can simply work in mm everywhere in the script, and
                then specify a scale value of .1 in the call to create_occurrence.
        Returns: The `Occurrence` that was created in the Fusion 360 document.
        """
        return self._create_occurrence(
            parent_occurrence=None,
            hidden=False,
            create_children=create_children,
            scale=scale)

    def _create_occurrence(self, parent_occurrence=None, hidden=True, create_children=True, scale=1):
        if scale != 1:
            return self.copy().scale(scale, scale, scale)._create_occurrence(
                parent_occurrence, hidden, create_children, 1)

        if not parent_occurrence:
            parent_component = root()
        else:
            parent_component = parent_occurrence.component
        occurrence = self._create_component(parent_component)
        occurrence.isLightBulbOn = not hidden
        if create_children:
            self._create_children(occurrence)
        for name in self._named_points.keys():
            construction_point_input = occurrence.component.constructionPoints.createInput()
            construction_point_input.setByPoint(self.named_point(name).point)
            construction_point = occurrence.component.constructionPoints.add(construction_point_input)
            construction_point.name = name
        return occurrence

    def _create_component(self, parent_component):
        return _create_component(
            parent_component, *self.bodies, name=self.name or self._default_name())

    def _create_children(self, occurrence):
        for child in self.children():
            child._create_occurrence(occurrence)

    def place(self, x: _vector_like = _null_vector, y: _vector_like = _null_vector,
              z: _vector_like = _null_vector):
        """Moves this component by the individual axes component of each of the 3 specified vectors.

        This is a powerful method that can be used in various ways to specify the location of a component.

        It is typically used to place this Component at some position relative to another Component or other
        BoundedEntity. This takes advantage of the __neg__ (-), __pos__ (+) and __invert__ (~) operator overrides on
        BoundedEntity that return a Place object, and also the __eq__ (==) operator override in the Place object.

        '-' is used to denote the minimum value of the entity in one of the 3 axes, '+' is used to denote the maximum,
        and '~' is used to denote the midpoint.

        You can leave out any of the placement components and this Component won't be moved along that axis.

        For example, to place this Component so that the minimum x, y and z points are aligned with the midpoint of
        another object, you could do::

            component.place(-component == ~other_component,
                            -component == ~other_component,
                            -component == ~other_component)

        or a slightly more complex example::

            component.place(-component == ~other_component,
                            +component == -other_component,
                            ~component == +other_component)

        This would place this component so that the minimum x bound of this object is aligned with the mid X point of
        the other object, the maximum y bound with the negative y bound, and the mid Z point with the maximum Z bound.

        You can also specify an offset to the alignment::

            component.place((-component == ~other_component) + 10,
                            +component == -other_component,
                            ~component == +other_component)

        This is the same as the previous example, except that the component is translated by an additional 10 cm in the
        x axis. Note that due to operator precedence, parenthesis are required around the == statement in this case.

        It can also be occasionally useful to specify an alignment based on some different object::

            component.place((-other_component == +some_other_component))

        And finally, you can also use this to specify a specific numeric location for any of the major axes::

            component.place(~component == 3, -component == 10, +component == -233.4471)

        Args:
            x: This Component will be translated by the x component of this vector.
            y: This Component will be translated by the y component of this vector.
            z: This Component will be translated by the z component of this vector.

        Returns: `self`
        """
        transform = Matrix3D.create()
        transform.translation = Vector3D.create(x.x, y.y, z.z)
        self._local_transform.transformBy(transform)
        self._reset_cache()
        return self

    def _reset_cache(self):
        super()._reset_cache()
        self._cached_brep_bodies = None
        self._cached_world_transform = None
        self._cached_inverse_transform = None
        for component in self.children():
            component._reset_cache()

    def world_transform(self) -> Matrix3D:
        if self._cached_world_transform is not None:
            return self._cached_world_transform.copy()
        transform = self._local_transform.copy()
        if self.parent is not None:
            transform.transformBy(self.parent.world_transform())
        self._cached_world_transform = transform
        return transform.copy()

    def _inverse_world_transform(self):
        if self._cached_inverse_transform is None:
            self._cached_inverse_transform = self.world_transform()
            self._cached_inverse_transform.invert()
        return self._cached_inverse_transform

    def get_plane(self) -> Optional[adsk.core.Plane]:
        """Returns: The plane that this Component lies in, or None if this is not a Planar component."""
        return None

    def transform(self, matrix: Matrix3D):
        """Transforms this component with the given transform.

        Note: matrices that produce a non-uniform scale will not work correctly.

        Args:
            matrix: The matrix transform

        Returns: `self`
        """
        transform = self._local_transform
        transform.transformBy(matrix)
        self._reset_cache()
        return self

    def rotate(self, rx: float = 0, ry: float = 0, rz: float = 0,
               center: Onion[Iterable[Onion[float, int]], Point3D] = None) -> 'Component':
        """Rotates this Component.

        The component will first be rotated around the X axis, then the Y axis, then the Z axis.

        Args:
            rx: The angle in degrees to rotate this object around the X axis by.
            ry: The angle in degrees to rotate this object around the Y axis by.
            rz: The angle in degrees to rotate this object around the Z axis by.
            center: If given, the rotation will occur around an axis parallel with each of the 3 major axes that run
                through this point.
        Returns: `self`
        """
        transform = self._local_transform

        if center is None:
            center_point = self._origin
        elif isinstance(center, Point3D):
            center_point = center
        elif isinstance(center, Point):
            center_point = center.point
        else:
            center_coordinates = list(center)[0:3]
            while len(center_coordinates) < 3:
                center_coordinates.append(0)
            center_point = Point3D.create(*center_coordinates)

        if rx != 0:
            rotation = Matrix3D.create()
            rotation.setToRotation(math.radians(rx), self._pos_x, center_point)
            transform.transformBy(rotation)
        if ry != 0:
            rotation = Matrix3D.create()
            rotation.setToRotation(math.radians(ry), self._pos_y, center_point)
            transform.transformBy(rotation)
        if rz != 0:
            rotation = Matrix3D.create()
            rotation.setToRotation(math.radians(rz), self._pos_z, center_point)
            transform.transformBy(rotation)
        self._reset_cache()
        return self

    def rx(self, angle: float, center: Onion[Iterable[Onion[float, int]], Point3D] = None) -> 'Component':
        """Rotates this Component around the X axis.

        Args:
            angle: The angle in degrees to rotate this object by.
            center: If given, the rotation will occur around an axis parallel with the X axis that runs through this
                point.

        Returns: `self`
        """
        return self.rotate(angle, center=center)

    def ry(self, angle: float, center: Onion[Iterable[Onion[float, int]], Point3D] = None) -> 'Component':
        """Rotates this Component around the Y axis.

        Args:
            angle: The angle in degrees to rotate this object by.
            center: If given, the rotation will occur around an axis parallel with the Y axis that runs through this
                point.

        Returns: `self`
        """
        return self.rotate(ry=angle, center=center)

    def rz(self, angle: float, center: Onion[Iterable[Onion[float, int]], Point3D] = None) -> 'Component':
        """Rotates this Component around the Z axis.

        Args:
            angle: The angle in degrees to rotate this object by.
            center: If given, the rotation will occur around an axis parallel with the Z axis that runs through this
                point.

        Returns: `self`
        """
        return self.rotate(rz=angle, center=center)

    def translate(self, tx: float = 0, ty: float = 0, tz: float = 0) -> 'Component':
        """Translates this Component the given distances along the 3 major axes.

        Args:
            tx: The distance to move this Component in the X axis.
            ty: The distance to move this Component in the Y axis.
            tz: The distance to move this Component in the Z axis.

        Returns: `self`
        """
        translation = Matrix3D.create()
        translation.translation = adsk.core.Vector3D.create(tx, ty, tz)
        self._local_transform.transformBy(translation)
        self._reset_cache()
        return self

    def tx(self, tx: float) -> 'Component':
        """Translates this Component the given distance in the X axis.

        Args:
            tx: The distance to move this Component in the X axis.

        Returns: `self`
        """
        return self.translate(tx)

    def ty(self, ty: float) -> 'Component':
        """Translates this Component the given distance in the Y axis.

        Args:
            ty: The distance to move this Component in the Y axis.

        Returns: `self`
        """
        return self.translate(ty=ty)

    def tz(self, tz: float) -> 'Component':
        """Translates this Component the given distance in the Z axis.

        Args:
            tz: The distance to move this Component in the Z axis.

        Returns: `self`
        """
        return self.translate(tz=tz)

    def scale(self, sx: float = 1, sy: float = 1, sz: float = 1,
              center: Onion[Iterable[Onion[float, int]], Point3D] = None) -> 'Component':
        """Uniformly scales this object.

        A mirror along one of the 3 major axes can also be achieved by specifying a negative scale factor, but the
        absolute value of all scale factors must be equal.

        Args:
            sx: The scale factor along the X axis
            sy: The scale factor along the Y axis
            sz: The scale factor along the Z axis
            center: The center of the scale operation. If not specified, the center will be the origin.

        Returns: `self`
        """
        scale = Matrix3D.create()
        translation = Matrix3D.create()
        if abs(sx) != abs(sy) or abs(sy) != abs(sz):
            raise ValueError("Non-uniform scaling is not currently supported")

        if center is None:
            center_point = self._origin
        elif isinstance(center, Point3D):
            center_point = center
        elif isinstance(center, Point):
            center_point = center.point
        else:
            center_coordinates = list(center)[0:3]
            while len(center_coordinates) < 3:
                center_coordinates.append(0)
            center_point = Point3D.create(*center_coordinates)

        translation.translation = center_point.asVector()
        translation.invert()
        self._local_transform.transformBy(translation)

        scale.setCell(0, 0, sx)
        scale.setCell(1, 1, sy)
        scale.setCell(2, 2, sz)
        self._local_transform.transformBy(scale)

        translation.invert()
        self._local_transform.transformBy(translation)

        self._reset_cache()
        return self

    @property
    def edges(self) -> Sequence[Edge]:
        """Returns: A Sequence of all edges associated with this component."""
        result = []
        for body in self.bodies:
            for edge in body.edges:
                result.append(edge)
        return result

    @property
    def faces(self) -> Sequence[Face]:
        """Returns: A Sequence of all faces associated with this component."""
        result = []
        for body in self.bodies:
            for face in body.faces:
                result.append(face)
        return result

    def find_faces(self, selector: _face_selector_types) -> Sequence[Face]:
        """Finds any faces that is coincident with any face in the given entities.

        This finds any face in this Component whose intersection with one of the faces in the face selectors is also a
        face. i.e. two faces that intersect in a point or a curve are not considered coincident.

        Args:
            selector: The entities used to find any coincident faces of in this Component

        Returns: A Sequence of the Faces that are coincident with one of the selector's Faces, or an empty Sequence if
            there are no such faces.
        """
        selector_faces = _flatten_face_selectors(selector)
        result = []
        for body in self.bodies:
            for face in _find_coincident_faces_on_body(body.brep, selector_faces):
                result.append(body.faces[_face_index(face)])
        return result

    def add_named_point(self, name: str, point: Onion[Sequence[float], Point3D, Point]):
        """Adds a point to this Component with the given name.

        The added point will remain in the same relative position in the component even when the component is
        moved/transformed.

        Args:
            name: The name of the point to add.
            point: A point to add to this Component. This may be any arbitrary point, it doesn't have to be an object
                (vertex, etc.) already associated with this Component.
        """
        if isinstance(point, Point):
            point = point.point
        if isinstance(point, Point3D):
            point = point.copy()
        else:
            point = Point3D.create(*point)
        point.transformBy(self._inverse_world_transform())
        self._named_points[name] = point

    def named_point(self, name) -> Optional[Point]:
        """Gets the Point in this Component with the given name.

        Args:
            name: The name of the Point to get

        Returns: The point with the given name, or None if no such point exists in this Component.
        """
        point = self._named_points.get(name)
        if not point:
            return None
        point = point.copy()
        point.transformBy(self.world_transform())
        return Point(point)

    def _find_face_index(self, face: Face) -> Tuple[int, int]:
        face_index = _face_index(face)
        body_index = _body_index(face.body, self.bodies)
        if body_index is None:
            raise ValueError("Could not find face in component")
        return body_index, face_index

    def add_named_faces(self, name: str, *faces: Face):
        """Associates a name with the specified Faces in this Component.

        The Faces can later be looked up by name using `named_faces(name)`

        Args:
            name: The name to associate with the given Faces.
            *faces: The faces to associate a name with. These must be Faces within this Component.
        """
        face_index_list = self._named_faces.get(name) or []
        for face in faces:
            face_index_list.append(self._find_face_index(face))
        self._named_faces[name] = face_index_list

    def named_faces(self, name) -> Optional[Sequence[Face]]:
        """Gets all faces with the specified name in this Component.

        Args:
            name: The name of the face

        Returns: A Sequence of Faces, or None if no Faces with the given name were found.
        """
        face_index_list = self._named_faces.get(name)
        if face_index_list is None:
            return None
        result = []
        for face_index in face_index_list:
            result.append(self.bodies[face_index[0]].faces[face_index[1]])
        return result

    def all_face_names(self) -> Sequence[str]:
        return list(self._named_faces.keys())

    def _find_edge_index(self, edge: Edge) -> Tuple[int, int]:
        edge_index = _edge_index(edge)
        body_index = _body_index(edge.body, self.bodies)
        if body_index is None:
            raise ValueError("Could not find edge in component")
        return body_index, edge_index

    def add_named_edges(self, name: str, *edges: Edge):
        """Associates a name with the specified Edges in this Component.

        The Edges can later be looked up by name using `named_edges(name)`

        Args:
            name: The name to associate with the given Edges.
            *edges: The edges to associate a name with. These must be Edges within this Component.
        """
        edge_index_list = self._named_edges.get(name) or []
        for edge in edges:
            edge_index_list.append(self._find_edge_index(edge))
        self._named_edges[name] = edge_index_list

    def named_edges(self, name) -> Optional[Sequence[Edge]]:
        """Gets all edges with the specified name in this Component.

        Args:
            name: The name of the edge

        Returns: A Sequence of Edges, or None if no Edges with the given name were found.
        """
        edge_index_list = self._named_edges.get(name)
        if edge_index_list is None:
            return None
        result = []
        for edge_index in edge_index_list:
            result.append(self.bodies[edge_index[0]].edges[edge_index[1]])
        return result

    def find_edges(self, selector: _edge_selector_types) -> Sequence[Edge]:
        """Finds any edges that is coincident with any face or edge in the given entities.

        This finds any edge in this Component that is coincident with any of the given selectors

        Args:
            selector: The entities used to find any coincident edges of in this Component

        Returns: A Sequence of the Edges that are coincident with one of the selector's Faces or Edges, or an empty
            Sequence if there are no such edges.
        """
        selectors = _flatten_edge_selectors(selector)
        result = []
        for body in self.bodies:
            for edge in _find_coincident_edges_on_body(body.brep, selectors):
                result.append(body.edges[_edge_index(edge)])
        return result

    def shared_edges(self, face_selector1: _face_selector_types,
                     face_selector2: _face_selector_types) -> Sequence[Edge]:
        """Finds any shared edges between any Face matching face_selector1, and any Face matching face_selector2.

        Args:
            face_selector1: A set of face selectors for the first set of faces
            face_selector2: A set of face selectors for the second set of faces

        Returns: All edges that are shared between the 2 sets of faces.
        """
        other_faces = self.find_faces(face_selector2)

        result_edges = []
        for face in self.find_faces(face_selector1):
            for edge in face.edges:
                for other_face in edge.faces:
                    if other_face != face and other_face in other_faces:
                        if edge not in result_edges:
                            result_edges.append(edge)
        return result_edges

    def closest_points(self, entity: _entity_types) -> Tuple[Point3D, Point3D]:
        """Finds the points on this entity and the specified entity that are closest to one another.

        In the case of parallel faces or other cases where there may be multiple sets of points at the same
        minimum distance, the exact points that are returned are unspecified.

        Args:
            entity: The entity to find the closest points with.

        Returns: A tuple of 2 Point3D objects. The first will be a point on this Component, and the second will be
            a point on the other entity.
        """
        self_body = _union_entities(self.bodies)
        other_body = _union_entities(entity)

        occ1 = _create_component(root(), self_body, name="temp")
        occ2 = _create_component(root(), other_body, name="temp")

        try:
            result = app().measureManager.measureMinimumDistance(occ1.bRepBodies[0], occ2.bRepBodies[0])
            return result.positionOne, result.positionTwo
        finally:
            occ1.deleteMe()
            occ2.deleteMe()

    def align_to(self, entity: _entity_types, vector: Vector3D) -> 'Component':
        """Moves this component along the given vector until it touches the specified entity.

        This uses a potentially iterative method based on the shortest distant between the 2 entities. In some cases,
        this may take many iterations to complete. For example, if there is a section where there are 2 parallel faces
        that are sliding past each other with a very small space between them.

        Args:
            entity: The entity representing the end point of the movement. The movement will stop once the object is
                within Fusion 360's point tolerance of this entity(`Application.pointTolerance`). It is guaranteed that
                this object will be exactly touching, or almost touching (within the above tolerance), but will *not*
                be "past" touching this entity.
            vector: The vector to move this component along

        Returns: `self`

        Raises:
            ValueError: If the entities do not intersect along the given vector. If this occurs, this Component will
                remain in its original position.
        """
        self_body = _union_entities(self.bodies)
        other_body = _union_entities(entity, vector=vector)
        axis = vector.copy()
        axis.normalize()
        other_axis = _get_arbitrary_perpendicular_unit_vector(vector)

        self_obb = app().measureManager.getOrientedBoundingBox(self_body, axis, other_axis)
        other_obb = app().measureManager.getOrientedBoundingBox(other_body, axis, other_axis)
        self_bb = _oriented_bounding_box_to_bounding_box(self_obb)
        other_bb = _oriented_bounding_box_to_bounding_box(other_obb)

        if not self_bb.intersects(other_bb):
            self_center = BoundingBox(self_bb).mid()

            other_min_point = Point3D.create(other_bb.minPoint.x,
                                             self_center.y,
                                             self_center.z)
            other_max_point = Point3D.create(other_bb.maxPoint.x,
                                             self_center.y,
                                             self_center.z)

            if other_bb.minPoint.x > self_bb.maxPoint.x:
                self_bb.expand(other_min_point)
            if other_bb.maxPoint.x > self_bb.maxPoint.x:
                self_bb.expand(other_max_point)
            if not self_bb.intersects(other_bb):
                raise ValueError("The 2 entities do not intersect along the given vector")

        occ1 = _create_component(root(), self_body, name="temp")
        occ2 = _create_component(root(), other_body, name="temp")

        try:
            while True:
                shortest_distance = app().measureManager.measureMinimumDistance(
                    occ1.bRepBodies[0], occ2.bRepBodies[0]).value
                if shortest_distance < app().pointTolerance:
                    break

                translation = axis.copy()
                translation.scaleBy(shortest_distance)
                translation_matrix = Matrix3D.create()
                translation_matrix.translation = translation
                transform = occ1.transform
                transform.transformBy(translation_matrix)
                occ1.transform = transform

                self_obb = app().measureManager.getOrientedBoundingBox(occ1.bRepBodies[0], vector, other_axis)
                self_bb = _oriented_bounding_box_to_bounding_box(self_obb)

                if self_bb.minPoint.x > other_bb.maxPoint.x:
                    raise ValueError("The 2 entities do not intersect along the given vector")

            self.translate(*occ1.transform.translation.asArray())
        finally:
            occ1.deleteMe()
            occ2.deleteMe()

        return self

    def thickness(self, axis):
        """Gets the maximum thickness of the component in the specified axis.

        Args:
            axis: A vector representing the axis to find the maximum thickness in.

        Returns: The maximum thickness of the component in the specified axis.
        """
        self_body = _union_entities(self.bodies)

        axis = axis.copy()
        axis.normalize()
        other_axis = _get_arbitrary_perpendicular_unit_vector(axis)

        return app().measureManager.getOrientedBoundingBox(self_body, axis, other_axis).length

    def oriented_bounding_box(
            self, x_axis: Vector3D, y_axis: Optional[Vector3D] = None, name: str = None) -> 'Box':
        """Returns a bounding box oriented such that it's x and y axis' are the given vectors.

        :param x_axis: The 'x' axis of the bounding box.
        :param y_axis: The 'y' axis of the bounding box. This must be perpendicular to the x axis vector. This can
        also be omitted, in which case a perpendicular vector will be chosen arbitrarily.
        :return: A Box object oriented and located as the oriented bounding box
        """
        x_axis = x_axis.copy()
        x_axis.normalize()
        if not y_axis:
            y_axis = _get_arbitrary_perpendicular_unit_vector(x_axis)
        else:
            y_axis = y_axis.copy()
            y_axis.normalize()

        self_body = _union_entities(self.bodies)
        bounding_box = app().measureManager.getOrientedBoundingBox(self_body, x_axis, y_axis)

        box = Box(
            bounding_box.length,
            bounding_box.width,
            bounding_box.height,
            name=name or "OrientedBoundingBox")

        z_axis = x_axis.crossProduct(y_axis)
        matrix = Matrix3D.create()
        matrix.setToAlignCoordinateSystems(
            Point3D.create(bounding_box.length / 2, bounding_box.width / 2, bounding_box.height/2),
            Vector3D.create(1, 0, 0),
            Vector3D.create(0, 1, 0),
            Vector3D.create(0, 0, 1),
            bounding_box.centerPoint,
            x_axis,
            y_axis,
            z_axis)

        box.transform(matrix)
        return box


class ComponentWithChildren(Component, ABC):
    def __init__(self, name):
        super().__init__(name)
        self._children = []

    def _add_children(self, children: Iterable[Component], func: Callable[[Component], None] = None):
        for child in children:
            if child.parent is not None:
                child = child.copy()
            child._local_transform.transformBy(self._inverse_world_transform())
            child._reset_cache()
            if func:
                func(child)
            child._parent = self
            self._children.append(child)
        self._reset_cache()

    def children(self) -> Sequence['Component']:
        return tuple(self._children)

    def _copy_to(self, copy: 'ComponentWithChildren', copy_children: bool):
        copy._cached_inverse_transform = None
        copy._children = []
        if copy_children:
            copy._add_children([child.copy() for child in self._children])

    def find_children(self, name, recursive=True) -> Sequence[Component]:
        result = []
        for child in self._children:
            if child.name == name:
                result.append(child)
        if recursive:
            for child in self._children:
                if isinstance(child, ComponentWithChildren):
                    result.extend(child.find_children(name, recursive))
        return result


class Shape(ComponentWithChildren):
    def __init__(self, *bodies: BRepBody, name: str):
        super().__init__(name)
        self._bodies = list(bodies)

    def _raw_bodies(self):
        return self._bodies

    def _copy_to(self, copy: 'Shape', copy_children: bool):
        super()._copy_to(copy, copy_children)
        copy._bodies = [brep().copy(body) for body in self._bodies]


class BRepComponent(Shape):
    """Defines a Component for a set of raw BRepBody or BrepFace objects.

    This can be useful when you need some advanced feature of Fusion's API that isn't supported by fscad, and can be
    used to wrap the result of that advanced operation and let it be used within the fscad API.

    Args:
        *brep_entities: The BRepBody or BRepFace objects that this Component will contain
        component: If provided, the component will be added as a hidden child of this component. e.g. this can be used
          to add the component the face comes from as a child
        name: The name of the Component
    """
    def __init__(self, *brep_entities: Onion[BRepBody, BRepFace], component: Component = None, name: str = None):
        super().__init__(*[brep().copy(brep_entity) for brep_entity in brep_entities], name=name)
        if component:
            self._add_children((component,))

    def get_plane(self) -> Optional[adsk.core.Plane]:
        plane = None
        for body in self.bodies:
            for face in body.faces:
                if not isinstance(face.brep.geometry, adsk.core.Plane):
                    return None
                if plane is None:
                    plane = face.brep.geometry
                elif not plane.isCoPlanarTo(face.brep.geometry):
                    return None
        return plane


class PlanarShape(Shape):
    def __init__(self, body: BRepBody, name: str):
        super().__init__(body, name=name)

    def get_plane(self) -> adsk.core.Plane:
        return self.bodies[0].faces[0].brep.geometry


class Box(Shape):
    """Defines a box.

    Args:
            x: The size of the box in the x axis
            y: The size of the box in the y axis
            z: The size of the box in the z axis
            name: The name of the Component
    """

    _top_index = 0
    _bottom_index = 1
    _front_index = 2
    _left_index = 3
    _back_index = 4
    _right_index = 5

    def __init__(self, x: float, y: float, z: float, name: str = None):
        body = brep().createBox(OrientedBoundingBox3D.create(
            Point3D.create(x/2, y/2, z/2),
            self._pos_x, self._pos_y,
            x, y, z))
        super().__init__(body, name=name)

    @property
    def top(self) -> Face:
        """Returns: The top Face of the box. i.e. The Face in the positive Z direction."""
        return self.bodies[0].faces[self._top_index]

    @property
    def bottom(self) -> Face:
        """Returns: The bottom Face of the box. i.e. the Face in the negative Z direction."""
        return self.bodies[0].faces[self._bottom_index]

    @property
    def left(self) -> Face:
        """Returns: The left Face of the box. i.e. the Face in the negative X direction."""
        return self.bodies[0].faces[self._left_index]

    @property
    def right(self) -> Face:
        """Returns: The right Face of the box. i.e. the Face in the positive X direction."""
        return self.bodies[0].faces[self._right_index]

    @property
    def front(self) -> Face:
        """Returns: The front Face of the box. i.e. the Face in the negative Y direction."""
        return self.bodies[0].faces[self._front_index]

    @property
    def back(self) -> Face:
        """Returns: The back face of the box. i.e. the Face in the positive Y direction."""
        return self.bodies[0].faces[self._back_index]


class Cylinder(Shape):
    """Defines a cylinder, a cone or a truncated cone.

    Args:
        height: The height of the cylinder
        radius: The radius of the cylinder. If top_radius is also specified, this is the radius of the bottom of the
            cylinder/cone.
        top_radius: If provided, the radius of the top of the cylinder/cone
        name: The name of the component
    """
    _side_index = 0

    def __init__(self, height: float, radius: float, top_radius: float = None, name: str = None):
        if radius == 0:
            # The API doesn't support the bottom radius being 0, so we have to swap the top and bottom points
            body = brep().createCylinderOrCone(Point3D.create(0, 0, height), top_radius, self._origin, radius)

        else:
            body = brep().createCylinderOrCone(self._origin, radius, Point3D.create(0, 0, height),
                                               top_radius if top_radius is not None else radius)
        if radius == 0:
            self._bottom_index = None
            self._top_index = 1
            self._reversed_axis = True
        elif top_radius == 0:
            self._bottom_index = 1
            self._top_index = None
            self._reversed_axis = False
        else:
            self._bottom_index = 1
            self._top_index = 2
            self._reversed_axis = False

        self._bottom_radius = radius
        if top_radius is None:
            self._top_radius = radius
        else:
            self._top_radius = top_radius

        self._height = height

        super().__init__(body, name=name)

    def _copy_to(self, copy: 'Cylinder', copy_children: bool):
        super()._copy_to(copy, copy_children)
        copy._bottom_index = self._bottom_index
        copy._top_index = self._top_index
        copy._bottom_radius = self._bottom_radius
        copy._top_radius = self._top_radius
        copy._height = self._height
        copy._reversed_axis = self._reversed_axis

    @property
    def top(self) -> Optional[Face]:
        """Returns: The top face of the cylinder/cone. May be None for a cone with a top radius of 0."""
        if self._top_index is None:
            return None
        return self.bodies[0].faces[self._top_index]

    @property
    def bottom(self) -> Optional[Face]:
        """Returns: The bottom Face of the cylinder/cone. May be None for a cone with a bottom radius of 0."""
        if self._bottom_index is None:
            return None
        return self.bodies[0].faces[self._bottom_index]

    @property
    def side(self) -> Face:
        """Returns: The side Face of the cylinder/cone."""
        return self.bodies[0].faces[self._side_index]

    @property
    def radius(self) -> float:
        return self._bottom_radius

    @property
    def bottom_radius(self) -> float:
        return self._bottom_radius

    @property
    def top_radius(self) -> float:
        return self._top_radius

    @property
    def height(self) -> float:
        return self._height

    @property
    def axis(self) -> Vector3D:
        axis: Vector3D = self.side.brep.geometry.axis
        if (self._reversed_axis):
            axis.scaleBy(-1)
        return axis

    @property
    def angle(self) -> float:
        """Returns: the angle of the side wall in degrees. A positive angle has a larger bottom radius."""
        return math.degrees(math.atan((self.bottom_radius - self.top_radius) / self.height))


class Sphere(Shape):
    """Defines a sphere.

    Args:
        radius: The radius of the sphere
        name: The name of the component
    """
    def __init__(self, radius: float, name: str = None):
        super().__init__(brep().createSphere(self._origin, radius), name=name)

    @property
    def surface(self) -> Face:
        """Returns: the Face representing the surface of the sphere."""
        return self.bodies[0].faces[0]


class Torus(Shape):
    """Defines a torus.

    Args:
        major_radius: The major radius of the torus
        minor_radius: The minor radius of the torus
        name: The name of the component
    """
    def __init__(self, major_radius: float, minor_radius: float, name: str = None):
        super().__init__(brep().createTorus(self._origin, Vector3D.create(0, 0, 1), major_radius, minor_radius),
                         name=name)

    @property
    def surface(self) -> Face:
        """Returns: the Face representing the surface of the torus."""
        return self.bodies[0].faces[0]


class Rect(PlanarShape):
    """Defines a 2D rectangle.

    Args:
        x: The size of the rectangle in the x axis
        y: The size of the rectangle in the y axis
        name: The name of the component
    """
    def __init__(self, x: float, y: float, name: str = None):
        # this is a bit faster than creating it from createWireFromCurves -> createFaceFromPlanarWires
        box = brep().createBox(OrientedBoundingBox3D.create(
            Point3D.create(x/2, y/2, -.5),
            self._pos_x, self._pos_y,
            x, y, 1))
        super().__init__(brep().copy(box.faces[Box._top_index]), name)


class Circle(PlanarShape):
    """Defines a 2D circle.

    Args:
        radius: The radius of the circle
        name: The name of the component
    """
    _top_index = 2

    def __init__(self, radius: float, name: str = None):
        # this is a bit faster than creating it from createWireFromCurves -> createFaceFromPlanarWires
        cylinder = brep().createCylinderOrCone(
            Point3D.create(0, 0, -1), radius, self._origin, radius)
        super().__init__(brep().copy(cylinder.faces[self._top_index]), name)


class Builder2D(object):
    """Builds a Planar 2D face in the x/y plane from a sequence of edge movements.

    Each call to one of the edge directives implicitly uses the ending point of the previous edge as the starting point
    of the next edge. All points should be "2d" - with either 2 components, or 3 components but with z=0.

    Args:
        start_point: The starting point of the face being built. The next edge directive will be relative to this point.
        name: The name of the component
    """

    _segments: List[Curve3D]

    def __init__(self, start_point: Onion[Tuple[float, float], Point2D, Point3D, Point], name: str = None):
        self._start_point = _point_2d(start_point)
        self._segments = []

    @property
    def first_point(self):
        """Returns: The first point that was added."""
        return self._start_point

    @property
    def last_point(self):
        """Returns: The endpoint of the last edge that has been added."""
        if self._segments:
            return self._segments[-1].evaluator.getEndPoints()[2]
        else:
            return self._start_point

    def line_to(self, point: Onion[Tuple[float, float], Point2D, Point3D, Point]):
        """Define a line from the previous end point to the given point.

        Args:
            point: The end point of the new line.
        """
        start_point = self.last_point
        end_point = _point_2d(point)
        self._segments.append(Line3D.create(start_point, end_point))

    def fit_spline_through(self, *points: List[Onion[Tuple[float, float], Point2D, Point3D, Point]]):
        """Create a fit point spline through the given points.

        The end point of the previously defined edge will be implicitly used as the first point of the spline.

        Args:
            points: The points that the smoothed spline should run through.
        """
        self._segments.append(_create_fit_point_spline(self.last_point, *points))

    def build(self, name=None):
        """Builds a Planar face in the x/y plane from the edges that have been defined on this builder.

        The edges should all be non-overlapping, and should form a closed loop.

        :param name: If given, the name of the component being made.
        :return: A Component containing the newly built face.
        """
        wire_body = brep().createWireFromCurves(self._segments, allowSelfIntersections=False)[0]
        return BRepComponent(brep().createFaceFromPlanarWires([wire_body]), name=name)


class Polygon(PlanarShape):
    """Defines an arbitrary 2D polygon.

    Args:
        *points: A sequence of vertices of the polygon. An edge will be added between each adjacent point in the
            sequence, and also one between the first and last point.
        name: The name of the component
    """
    def __init__(self, *points: Onion[Tuple[float, float], Point2D, Point3D, Point], name: str = None):
        lines = []
        for i in range(-1, len(points)-1):
            lines.append(adsk.core.Line3D.create(_point_3d(points[i]), _point_3d(points[i+1])))
        wire, _ = brep().createWireFromCurves(lines)
        super().__init__(brep().createFaceFromPlanarWires((wire,)), name)


class RegularPolygon(Polygon):
    """Defines a regular 2D polygon.

    Args:
        sides: The number of sides of the polygon
        radius: The "radius" of the polygon. This is normally the distance from the center to the midpoint of a side.
            Also known as the apothem or inradius. If is_outer_radius is True, this is instead the distance from the
            center to a vertex.
        is_outer_radius: Whether radius is specified as the apothem/inradius, or the outer radius
        name: The name of the component
    """
    def __init__(self, sides: int, radius: float, is_outer_radius: bool = True, name: str = None):
        step_angle = 360 / sides
        points = []
        if not is_outer_radius:
            if sides % 2 == 0:
                radius = radius / (math.cos(math.radians(180) / sides))
            else:
                radius = 2 * radius / (1 + math.cos(math.radians(180) / sides))

        for i in range(0, sides):
            angle = step_angle / 2 + step_angle * i
            points.append(Point3D.create(
                radius * math.sin(math.radians(angle)),
                -radius * math.cos(math.radians(angle)),
                0))
        super().__init__(*points, name=name)


def import_fusion_archive(filename, name="import"):
    """Imports the given fusion archive as a new Component

    Args:
        filename: The filename of the local fusion archive
        name: The name of the component

    Returns: A new Component containing the contents of the imported file.
    """
    import_options = app().importManager.createFusionArchiveImportOptions(filename)

    document = app().importManager.importToNewDocument(import_options)
    imported_root = document.products[0].rootComponent

    bodies = []

    for body in imported_root.bRepBodies:
        bodies.append(brep().copy(body))
    for occurrence in imported_root.allOccurrences:
        for body in occurrence.bRepBodies:
            bodies.append(brep().copy(body))

    document.close(saveChanges=False)

    return BRepComponent(*bodies, name=name)


def import_dxf(filename, name="import"):
    """Imports the given fusion archive as a new Component

    Args:
        filename: The filename of the local fusion archive
        name: The name of the component

    Returns: A new Component containing the contents of the imported file.
    """

    import_options = app().importManager.createDXF2DImportOptions(filename, root().xYConstructionPlane)

    result = app().importManager.importToTarget2(import_options, root())
    sketch = result[0]  # type: adsk.fusion.Sketch

    curves = []
    for profile_curve in sketch.profiles[0].profileLoops[0].profileCurves:
        curves.append(profile_curve.geometry)
    wire_body, _ = brep().createWireFromCurves(curves)
    face_silhouette = brep().createFaceFromPlanarWires([wire_body])

    for sketch in result:
        sketch.deleteMe()

    return BRepComponent(face_silhouette, name=name)


class Combination(ComponentWithChildren, ABC):
    def __init__(self, name):
        super().__init__(name)
        self._cached_plane = None
        self._cached_plane_populated = False

    def _reset_cache(self):
        super()._reset_cache()
        self._cached_plane = None
        self._cached_plane_populated = False

    def _copy_to(self, copy: 'ComponentWithChildren', copy_children: bool):
        super()._copy_to(copy, copy_children)
        copy._cached_plane = None
        copy._cached_plane_populated = False

    def _raw_plane(self) -> Optional[adsk.core.Plane]:
        raise NotImplementedError()

    def get_plane(self) -> Optional[adsk.core.Plane]:
        if self._cached_plane_populated:
            return self._cached_plane
        raw_plane = self._raw_plane()
        if raw_plane is None:
            self._cached_plane = None
            self._cached_plane_populated = True
        else:
            world_transform = self.world_transform()
            plane = raw_plane.copy()
            plane.transformBy(world_transform)
            self._cached_plane = plane
        return self._cached_plane


class Union(Combination):
    """Unions a number of Components together.

    Args:
        *components: The Components to union together. The Components must be all planar, or all non-planar.
        name: The name of the Component
    """

    def __init__(self, *components: Component, name: str = None):
        super().__init__(name)

        self._body = None
        self._plane = None

        def process_child(child: Component):
            for body in child.bodies:
                if self._body is None:
                    self._body = brep().copy(body.brep)
                    self._plane = child.get_plane()
                else:
                    self._check_coplanarity(child)
                    brep().booleanOperation(self._body, body.brep, adsk.fusion.BooleanTypes.UnionBooleanType)
        self._add_children(components, process_child)

    def _raw_bodies(self) -> Iterable[BRepBody]:
        return self._body,

    def _raw_plane(self) -> Optional[adsk.core.Plane]:
        return self._plane

    def _copy_to(self, copy: 'Union', copy_children: bool):
        copy._body = brep().copy(self._body)
        copy._plane = self._plane
        super()._copy_to(copy, copy_children)

    def _check_coplanarity(self, child):
        if self._body is not None:
            plane = self.get_plane()
            child_plane = child.get_plane()

            if (child_plane is None) ^ (plane is None):
                raise ValueError("Cannot union a planar entity with a 3d entity")
            if plane is not None and not plane.isCoPlanarTo(child_plane):
                raise ValueError("Cannot union planar entities that are non-coplanar")


class Difference(Combination):
    """Represents the difference between a Component and any number of other Components.

    Args:
        *components: The Components to perform a difference on. The first Component will be the "positive"
            Component, and any remaining Components will be subtracted from the first Component. If the first
            Component is a non-planar Component, the remaining Components must also be non-planar. However, it is
            valid to subtract a non-planar Component from a planar Component.
        name: The name of the Component
    """
    def __init__(self, *components: Component, name: str = None):
        super().__init__(name)
        self._bodies = None

        self._plane = None

        def process_child(child: Component):
            self._check_coplanarity(child)
            if self._bodies is None:
                self._bodies = [brep().copy(child_body.brep) for child_body in child.bodies]
                self._plane = child.get_plane()
            else:
                for target_body in self._bodies:
                    for tool_body in child.bodies:
                        brep().booleanOperation(target_body, tool_body.brep,
                                                adsk.fusion.BooleanTypes.DifferenceBooleanType)
        self._add_children(components, process_child)

    def _raw_bodies(self) -> Iterable[BRepBody]:
        return self._bodies

    def _raw_plane(self) -> Optional[adsk.core.Plane]:
        return self._plane

    def _copy_to(self, copy: 'Difference', copy_children: bool):
        copy._bodies = [brep().copy(body) for body in self._bodies]
        copy._plane = self._plane
        super()._copy_to(copy, copy_children)

    def _check_coplanarity(self, child):
        if self._bodies is not None and len(self._bodies) > 0:
            plane = self.get_plane()
            child_plane = child.get_plane()

            if plane is None:
                if child_plane is not None:
                    raise ValueError("Cannot subtract a planar entity from a 3d entity")
            else:
                if child_plane is not None and not plane.isCoPlanarTo(child_plane):
                    raise ValueError("Cannot subtract planar entities that are non-coplanar")


class Intersection(Combination):
    """Represents the intersection between multiple Components.

    Args:
        *components: The Components to intersect with each other. This can be a mix of planar and non-planar geometry.
        name: The name of the Component
    """
    def __init__(self, *components: Component, name: str = None):
        super().__init__(name)
        self._bodies = None
        self._plane = None

        def process_child(child: Component):
            self._reset_cache()
            self._plane = self._check_coplanarity(child, self._plane)
            if self._bodies is None:
                self._bodies = [brep().copy(child_body.brep) for child_body in child.bodies]
            else:
                for target_body in self._bodies:
                    for tool_body in child.bodies:
                        brep().booleanOperation(target_body, tool_body.brep,
                                                adsk.fusion.BooleanTypes.IntersectionBooleanType)
        self._add_children(components, process_child)

    def _raw_bodies(self) -> Iterable[BRepBody]:
        return self._bodies

    def _raw_plane(self) -> Optional[adsk.core.Plane]:
        return self._plane

    def _copy_to(self, copy: 'Difference', copy_children: bool):
        copy._bodies = [brep().copy(body) for body in self._bodies]
        copy._plane = self._plane
        super()._copy_to(copy, copy_children)

    def _check_coplanarity(self, child, plane):
        if self._bodies:
            child_plane = child.get_plane()
            if plane is not None:
                if child_plane is not None and not plane.isCoPlanarTo(child_plane):
                    raise ValueError("Cannot intersect planar entities that are non-coplanar")
                return plane
            elif child_plane is not None:
                return child_plane
        else:
            return child.get_plane()


class Group(Combination):
    """Groups a set of Components without performing any operation on them.

    This can be useful when you need to perform some operation on a group of Components, but don't want to Union or
    otherwise join them together.

    Args:
        visible_children: The children to add as visible children
        hidden_children: The children to add as hidden children
        name: The name of the Component
    """
    def __init__(self, visible_children, hidden_children=None, name=None):
        super().__init__(name)

        self._visible_children = []
        self._hidden_children = []
        self._raw_brep_bodies = []

        self._plane = None

        def process_visible_child(child: Component):
            child_plane = child.get_plane()
            if len(self._visible_children) == 0:
                self._plane = child_plane
            elif child_plane and self._plane:
                if not child_plane.isCoPlanarTo(self._plane):
                    self._plane = None

            for body in child.bodies:
                self._raw_brep_bodies.append(brep().copy(body.brep))

            self._visible_children.append(child)
        self._add_children(visible_children, process_visible_child)

        if hidden_children:
            def process_hidden_child(child: Component):
                child_plane = child.get_plane()
                if len(self._visible_children) + len(self._hidden_children) == 0:
                    self._plane = child_plane
                elif child_plane and self._plane:
                    if not child_plane.isCoPlanarTo(self._plane):
                        self._plane = None

                self._hidden_children.append(child)
            self._add_children(hidden_children, process_hidden_child)

    def _raw_bodies(self) -> Iterable[BRepBody]:
        return self._raw_brep_bodies

    def _raw_plane(self) -> Optional[adsk.core.Plane]:
        return self._plane

    def _copy_to(self, copy: 'Union', copy_children: bool):
        copy._raw_brep_bodies = [brep().copy(body) for body in self._raw_brep_bodies]
        copy._visible_children = []
        copy._hidden_children = []
        copy._plane = self._plane

        super()._copy_to(copy, copy_children=False)

        copy._visible_children = [
            child.copy() for child in self._visible_children]
        copy._add_children(copy._visible_children)

        if copy_children:
            copy._hidden_children = [
                child.copy() for child in self._hidden_children]
            copy._add_children(copy._hidden_children)

    def _create_occurrence(self, parent_occurrence=None, hidden=True, create_children=False, scale=1) -> adsk.fusion.Occurrence:
        if scale != 1:
            return self.copy().scale(scale, scale, scale)._create_occurrence(
                parent_occurrence, hidden, create_children, 1)

        if parent_occurrence:
            parent_component = parent_occurrence.component
        else:
            parent_component = root()

        occurrence = self._create_component(parent_component=parent_component)
        occurrence.isLightBulbOn = not hidden
        for child in self._visible_children:
            child._create_occurrence(occurrence, hidden=False, create_children=create_children, scale=1)
        if create_children:
            for child in self._hidden_children:
                child._create_occurrence(occurrence, hidden=True, create_children=create_children, scale=1)
        for name in self._named_points.keys():
            construction_point_input = occurrence.component.constructionPoints.createInput()
            construction_point_input.setByPoint(self.named_point(name).point)
            construction_point = occurrence.component.constructionPoints.add(construction_point_input)
            construction_point.name = name
        return occurrence

    def _create_component(self, parent_component):
        return _create_component(parent_component, name=self.name or self._default_name())

    def _create_children(self, occurrence):
        for child in self._visible_children:
            child._create_occurrence(occurrence, hidden=False)
        for child in self._hidden_children:
            child._create_occurrence(occurrence, hidden=True)


class Loft(ComponentWithChildren):
    """Represents a body created by lofting through a set of faces.

    Currently, only a basic loft is supported. Support for center lines and guide rails, etc. is not yet implemented.

    Args:
        *components: The Components to loft through. These must all be planar Components.
        name: The name of the Component
    """

    def __init__(self, *components: Component, name: str = None):
        super().__init__(name)
        loft_sections = []

        def process_child(child: Component):
            nonlocal loft_sections
            if child.get_plane() is None:
                raise ValueError("Only planar geometry can be used with Loft")

            component_face = None
            for child_body in child.bodies:
                for face in child_body.faces:
                    if component_face is None:
                        component_face = face
                    else:
                        raise ValueError("A loft section must have only 1 face")
            loft_sections.append(brep().copy(component_face.brep))

        self._add_children(components, process_child)

        occurrence = _create_component(root(), *loft_sections, name="loft_temp")
        loft_feature_input = occurrence.component.features.loftFeatures.createInput(
            adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        for body in occurrence.bRepBodies:
            loft_feature_input.loftSections.add(body.faces[0])
        loft_feature = occurrence.component.features.loftFeatures.add(loft_feature_input)
        self._bottom_index = _face_index(loft_feature.startFace)
        self._top_index = _face_index(loft_feature.endFace)
        self._body = brep().copy(loft_feature.bodies[0])
        occurrence.deleteMe()

    def _raw_bodies(self) -> Iterable[BRepBody]:
        return self._body,

    def _copy_to(self, copy: 'Loft', copy_children: bool):
        copy._body = brep().copy(self._body)
        super()._copy_to(copy, copy_children)

    @property
    def bottom(self) -> Face:
        """The bottom (starting) face of the loft."""
        return self.bodies[0].faces[self._bottom_index]

    @property
    def top(self) -> Face:
        """The top (ending) face of the loft."""
        return self.bodies[0].faces[self._top_index]

    @property
    def sides(self) -> Iterable[Face]:
        """All side faces of the loft."""
        side_faces = []
        for i in range(0, len(self.bodies[0].faces)):
            if i != self._bottom_index and i != self._top_index:
                side_faces.append(self.bodies[0].faces[i])
        return side_faces


class Revolve(ComponentWithChildren):
    """Revolves face about an axis.

    Args:
        entity: The object to revolve. This can be a Face, an iterable of Faces, or a planar Component.
        axis: The axis to rotate around
        angle: The angle to revolve, in degrees. If not provided, a full 360 is used.
        name: The name of the component
    """
    _bodies = ...  # type: Sequence[BRepBody]

    def __init__(self, entity: Onion[Component, Face, Iterable[Face]], axis: Onion[adsk.core.Line3D, Edge],
                 angle: float = 360.0, name: str = None):
        super().__init__(name)

        if isinstance(entity, Component):
            component = entity
            if component.get_plane() is None:
                raise ValueError("Can't revolve non-planar geometry with Revolve.")
            faces = []
            for body in component.bodies:
                faces.extend(body.brep.faces)
        elif isinstance(entity, Face):
            component = entity.component
            if entity.get_plane() is None:
                raise ValueError("Can't revolve non-planar geometry with Revolve.")
            faces = [entity.brep]
        elif isinstance(entity, Iterable):
            component = None
            faces = []
            plane: adsk.core.Plane = None
            for face in entity:
                if component is None:
                    component = face.component
                elif face.component != component:
                    raise ValueError("All faces must be from the same component")

                if face.get_plane() is None:
                    raise ValueError("Can't revolve non-planar geometry with Revolve.")
                if plane is None:
                    plane = face.get_plane()
                elif not plane.isCoPlanarTo(face.get_plane()):
                    raise ValueError("All faces to revolve must be coplanar.")
                faces.append(face)
        else:
            raise ValueError("Unsupported object type for revolve: %s" % entity.__class__.__name__)

        input_bodies = []
        for face in faces:
            if face.body not in input_bodies:
                input_bodies.append(face.body)
        temp_occurrence = _create_component(root(), *input_bodies, name="temp")

        axis_line = None
        if isinstance(axis, Edge):
            if not isinstance(axis.brep.geometry, adsk.core.Line3D):
                raise ValueError("Only linear edges may be used as an axis.")
            axis_line = axis.brep.geometry
        elif isinstance(axis, adsk.core.Line3D):
            axis_line = axis
        else:
            raise ValueError("Unsupported axis type for revolve: %s" % axis.__class__.__name__)

        axis_input = temp_occurrence.component.constructionAxes.createInput()
        axis_input.setByLine(
            adsk.core.InfiniteLine3D.create(axis_line.endPoint, axis_line.startPoint.vectorTo(axis_line.endPoint)))
        axis_value = temp_occurrence.component.constructionAxes.add(axis_input)

        temp_bodies = list(temp_occurrence.bRepBodies)

        temp_faces = []
        for face in faces:
            body_index = input_bodies.index(face.body)
            temp_body = temp_bodies[body_index]
            temp_faces.append(_map_face(face, temp_body))

        revolve_input = temp_occurrence.component.features.revolveFeatures.createInput(
            _collection_of(temp_faces),
            axis_value,
            adsk.fusion.FeatureOperations.JoinFeatureOperation)
        revolve_input.setAngleExtent(False, ValueInput.createByReal(math.radians(angle)))
        temp_occurrence.component.features.revolveFeatures.add(revolve_input)

        feature = temp_occurrence.component.features.revolveFeatures[
            temp_occurrence.component.features.revolveFeatures.count - 1]

        bodies = []
        for body in feature.bodies:
            bodies.append(brep().copy(body))
        self._bodies = bodies

        self._add_children((component,))

        temp_occurrence.deleteMe()

    def _raw_bodies(self) -> Iterable[BRepBody]:
        return self._bodies

    def _copy_to(self, copy: 'ComponentWithChildren', copy_children: bool):
        super()._copy_to(copy, copy_children)
        copy._bodies = list(self._bodies)


class Sweep(ComponentWithChildren):
    """Sweeps a planar face along a path, optionally with a twist.

    Args:
        entity: The object to twist. This can be a Face, an iterable of Faces, or a planar Component.
        path: The path to sweep the face along
        turns: The number of turns to twist
        name: The name of the component
    """
    _bodies = ...  # type: Sequence[BRepBody]

    def __init__(self,
                 entity: Onion[Component, Face, Iterable[Face]],
                 path: Sequence[Onion[adsk.core.Curve3D, Edge]],
                 turns: float = 0,
                 name: str = None):
        super().__init__(name)

        components = []

        if isinstance(entity, Component):
            component = entity
            components.append(component)
            if component.get_plane() is None:
                raise ValueError("Can't sweep non-planar geometry with Sweep.")
            faces = []
            for body in component.bodies:
                faces.extend(body.brep.faces)
        elif isinstance(entity, Face):
            component = entity.component
            components.append(component)
            if entity.get_plane() is None:
                raise ValueError("Can't sweep non-planar geometry with Sweep.")
            faces = [entity.brep]
        elif isinstance(entity, Iterable):
            component = None
            faces = []
            plane: adsk.core.Plane = None
            for face in entity:
                if component is None:
                    component = face.component
                elif face.component != component:
                    raise ValueError("All faces must be from the same component")

                if face.get_plane() is None:
                    raise ValueError("Can't sweep non-planar geometry with Sweep.")
                if plane is None:
                    plane = face.get_plane()
                elif not plane.isCoPlanarTo(face.get_plane()):
                    raise ValueError("All faces to sweep must be coplanar.")
                faces.append(face)
            components.append(component)
        else:
            raise ValueError("Unsupported object type for sweep: %s" % entity.__class__.__name__)

        input_bodies = []
        for face in faces:
            if face.body not in input_bodies:
                input_bodies.append(face.body)
        temp_occurrence = _create_component(root(), *input_bodies, name="temp")

        temp_bodies = list(temp_occurrence.bRepBodies)
        temp_faces = []
        for face in faces:
            body_index = input_bodies.index(face.body)
            temp_body = temp_bodies[body_index]
            temp_faces.append(_map_face(face, temp_body))

        construction_plane_input = temp_occurrence.component.constructionPlanes.createInput(temp_occurrence)
        construction_plane_input.setByPlane(adsk.core.Plane.create(
            Point3D.create(0, 0, 0),
            Vector3D.create(0, 0, 1)))
        construction_plane = temp_occurrence.component.constructionPlanes.add(construction_plane_input)
        sketch = temp_occurrence.component.sketches.add(construction_plane, temp_occurrence)

        edges = []
        for item in path:
            if isinstance(item, Edge):
                if item.component not in components:
                    components.append(item.component)
                wire = brep().copy(item.brep)
            elif isinstance(item, Curve3D):
                wire = _create_wire(item)
            else:
                raise ValueError("Unsupported axis type for sweep: %s" % item.__class__.__name__)

            wire = temp_occurrence.component.bRepBodies.add(wire)
            edges.append(sketch.include(wire.edges[0])[0])
            wire.deleteMe()

        path_object = temp_occurrence.component.features.createPath(
            _collection_of(edges),
            isChain=False)

        sweep_input = temp_occurrence.component.features.sweepFeatures.createInput(
            _collection_of(temp_faces),
            path_object,
            adsk.fusion.FeatureOperations.JoinFeatureOperation)

        sweep_input.distanceOne = ValueInput.createByReal(1.0)
        sweep_input.twistAngle = ValueInput.createByReal(turns * math.pi * 2)

        feature = temp_occurrence.component.features.sweepFeatures.add(sweep_input)

        bodies = []
        for body in feature.bodies:
            bodies.append(brep().copy(body))
        self._bodies = bodies

        self._add_children(components)

        temp_occurrence.deleteMe()

    def _raw_bodies(self) -> Iterable[BRepBody]:
        return self._bodies

    def _copy_to(self, copy: 'ComponentWithChildren', copy_children: bool):
        super()._copy_to(copy, copy_children)
        copy._bodies = list(self._bodies)


class ExtrudeBase(ComponentWithChildren):
    _bodies = ...  # type: Sequence[BRepBody]

    def __init__(self, component: Component, faces: Iterable[BRepFace], extent, name: str = None):
        super().__init__(name)

        input_bodies = []
        for face in faces:
            if face.body not in input_bodies:
                input_bodies.append(face.body)
        temp_occurrence = _create_component(root(), *input_bodies, name="temp")

        temp_bodies = list(temp_occurrence.bRepBodies)

        temp_faces = []
        for face in faces:
            body_index = input_bodies.index(face.body)
            temp_body = temp_bodies[body_index]
            temp_faces.append(_map_face(face, temp_body))

        extrude_input = temp_occurrence.component.features.extrudeFeatures.createInput(
            _collection_of(temp_faces), adsk.fusion.FeatureOperations.JoinFeatureOperation)
        extrude_input.setOneSideExtent(
            extent, adsk.fusion.ExtentDirections.PositiveExtentDirection, ValueInput.createByReal(0))
        temp_occurrence.component.features.extrudeFeatures.add(extrude_input)
        # extrudeFeatures.add sometimes returns a feature with the wrong body? wth?
        # Getting it by index seems to work at least.
        feature = temp_occurrence.component.features.extrudeFeatures[
            temp_occurrence.component.features.extrudeFeatures.count - 1]

        bodies = []
        feature_body_map = {}
        feature_bodies = list(feature.bodies)
        # In some cases, the face being extruded is included in the bodies for some reason. If so, we want to
        # exclude it.
        for i, body in enumerate(feature_bodies):
            if body.isSolid:
                feature_body_map[i] = len(bodies)
                bodies.append(brep().copy(body))
        self._bodies = bodies

        self._start_face_indices = []
        for face in feature.startFaces:
            body_index = feature_body_map.get(feature_bodies.index(face.body))
            # Exclude any faces that come from the input face
            if body_index is not None:
                self._start_face_indices.append((body_index, _face_index(face)))

        self._end_face_indices = []
        for face in feature.endFaces:
            body_index = feature_body_map.get(feature_bodies.index(face.body))
            # Exclude any faces that come from the input face
            if body_index is not None:
                self._end_face_indices.append((body_index, _face_index(face)))

        self._side_face_indices = []
        for face in feature.sideFaces:
            body_index = feature_body_map.get(feature_bodies.index(face.body))
            # Exclude any faces that come from the input face
            if body_index is not None:
                self._side_face_indices.append((body_index, _face_index(face)))

        self._add_children([component])

        self._cached_start_faces = None
        self._cached_end_faces = None
        self._cached_side_faces = None

        temp_occurrence.deleteMe()

    def _copy_to(self, copy: 'ComponentWithChildren', copy_children: bool):
        super()._copy_to(copy, copy_children)
        copy._bodies = list(self._bodies)
        copy._start_face_indices = list(self._start_face_indices)
        copy._end_face_indices = list(self._end_face_indices)
        copy._side_face_indices = list(self._side_face_indices)
        copy._cached_start_faces = None
        copy._cached_end_faces = None
        copy._cached_side_faces = None

    def _raw_bodies(self) -> Iterable[BRepBody]:
        return self._bodies

    def _reset_cache(self):
        super()._reset_cache()
        self._cached_start_faces = None
        self._cached_end_faces = None
        self._cached_side_faces = None

    def _get_faces(self, indices) -> Sequence[Face]:
        result = []
        for body_index, face_index in indices:
            result.append(self.bodies[body_index].faces[face_index])
        return result

    @property
    def start_faces(self) -> Sequence[Face]:
        """The faces of the resulting body that the extrude starts from.

        If extruding an existing face of a 3d object, this will normally be empty. If extruding a face or set of faces,
        this will normally be the starting faces that were used to perform the extrude.
        """
        if not self._cached_start_faces:
            self._cached_start_faces = self._get_faces(self._start_face_indices)
        return list(self._cached_start_faces)

    @property
    def end_faces(self) -> Sequence[Face]:
        """The faces of the resulting body that correspond with the end of the extrusion."""
        if not self._cached_end_faces:
            self._cached_end_faces = self._get_faces(self._end_face_indices)
        return list(self._cached_end_faces)

    @property
    def side_faces(self) -> Sequence[Face]:
        """The faces of the resulting body that correspond to the sides of the extrusion."""
        if not self._cached_side_faces:
            self._cached_side_faces = self._get_faces(self._side_face_indices)
        return list(self._cached_side_faces)


class Extrude(ExtrudeBase):
    """Extrudes faces a certain distance.

    Args:
        entity: The object to extrude. This can be a Face, an iterable of Faces, or a planar Component.
        height: The distance to extrude the faces
        name: The name of the component
    """
    def __init__(self, entity: Onion[Component, Face, Iterable[Face]], height: float, name: str = None):
        if isinstance(entity, Component):
            component = entity
            if component.get_plane() is None:
                raise ValueError("Can't extrude non-planar geometry with Extrude.")
            faces = []
            for body in component.bodies:
                faces.extend(body.brep.faces)
        elif isinstance(entity, Face):
            component = entity.component
            faces = [entity.brep]
        elif isinstance(entity, Iterable):
            component = None
            faces = []
            for face in entity:
                if component is None:
                    component = face.component
                elif face.component != component:
                    raise ValueError("All faces must be from the same component")
                faces.append(face.brep)
        else:
            raise ValueError("Unsupported object type for extrude: %s" % entity.__class__.__name__)
        super().__init__(
            component, faces, adsk.fusion.DistanceExtentDefinition.create(adsk.core.ValueInput.createByReal(height)),
            name)


class ExtrudeTo(ExtrudeBase):
    """Extrudes the specified faces until they intersect with the face of another object.

    Args:
        entity: The faces to extrude. This can be a Face, an Iterable of Faces or a planar Component.
        to_entity: The object to extrude the faces to
        name: The name of the component
    """
    def __init__(self, entity: Onion[Face, Component, Iterable[Face]],
                 to_entity: Onion[Component, Face, Body],
                 offset=0.0,
                 name: str = None):
        if isinstance(entity, Component):
            component = entity
            if entity.get_plane() is None:
                raise ValueError("Can't extrude non-planar geometry with Extrude.")
            faces = []
            for body in entity.bodies:
                faces.extend(body.brep.faces)
        elif isinstance(entity, Face):
            component = entity.component
            faces = [entity.brep]
        elif isinstance(entity, Iterable):
            component = None
            faces = []
            for face in entity:
                if component is None:
                    component = face.component
                elif face.component != component:
                    raise ValueError("All faces must be from the same component")
                faces.append(face.brep)
        else:
            raise ValueError("Unsupported object type for extrude: %s" % entity.__class__.__name__)

        if isinstance(to_entity, Component):
            bodies = to_entity.bodies
            if len(bodies) > 1:
                raise ValueError("If to_entity is a component, it must contain only a single body")
            component_to_add = to_entity.copy()
            temp_occurrence = to_entity.create_occurrence(False)
            to_entity = temp_occurrence.bRepBodies[0]
        elif isinstance(to_entity, Body):
            temp_occurrence = _create_component(root(), to_entity.brep, name="temp")
            component_to_add = to_entity.component
            to_entity = temp_occurrence.bRepBodies[0]
        elif isinstance(to_entity, Face):
            temp_occurrence = _create_component(root(), to_entity.body.brep, name="temp")
            component_to_add = to_entity.component
            to_entity = temp_occurrence.bRepBodies[0].faces[_face_index(to_entity)]
        else:
            raise ValueError("Unsupported type for to_entity: %s" % to_entity.__class__.__name__)

        super().__init__(component, faces, adsk.fusion.ToEntityExtentDefinition.create(
            to_entity, False, ValueInput.createByReal(offset)), name)
        self._add_children([component_to_add])
        temp_occurrence.deleteMe()


class OffsetEdges(ComponentWithChildren):
    """Offset the given edges on the given face by some amount.

    Args:
        face: A planar face containing the edges to offset
        edges: The edges to offset
        offset: How far to offset the edges. A positive grows the face, while a negative offset shrinks the face.
        name: The name of the component
    """
    def __init__(self, face: Face, edges: Sequence[Edge], offset: float, name: str = None):
        super().__init__(name)

        temp_occurrence = _create_component(root(), face.body, name="temp")

        temp_face = _map_face(face, temp_occurrence.bRepBodies[0])
        temp_body = temp_face.body

        sketch = temp_occurrence.component.sketches.add(temp_face)

        to_offset = []
        for edge in edges:
            to_offset.extend(sketch.include(temp_body.edges[_edge_index(edge.brep)]))

        offset_sketch_curves: Sequence[SketchCurve] = sketch.offset(
            _collection_of(to_offset), sketch.modelToSketchSpace(face.brep.pointOnFace), -offset)

        for curve in offset_sketch_curves:
            if hasattr(curve, "endSketchPoint"):
                if curve.endSketchPoint.connectedEntities.count == 1:
                    curve.extend(curve.endSketchPoint.geometry, createConstraints=False)
                if curve.startSketchPoint.connectedEntities.count == 1:
                    curve.extend(curve.startSketchPoint.geometry, createConstraints=False)

        start_edge, end_edge, loop = _find_connected_edge_endpoints(edges, face)

        new_face_edges = [sketch_curve.worldGeometry for sketch_curve in offset_sketch_curves]

        if not start_edge:
            # the original set of curves for a closed loop, so the offset curves should too
            for edge in edges:
                new_face_edges.append(edge.brep.geometry)
        else:
            # The original set of curves do not form a closed loop. The offset curves may or may not.
            offset_closed = False
            if len(offset_sketch_curves) == 1 and (isinstance(offset_sketch_curves[0], SketchCircle) or isinstance(offset_sketch_curves[0], SketchEllipse)):
                offset_closed = True
            elif len(offset_sketch_curves) > 1 and offset_sketch_curves[0].startSketchPoint == offset_sketch_curves[offset_sketch_curves.count - 1].endSketchPoint:
                offset_closed = True

            if offset_closed:
                # if the offset curve forms a closed loop, then we want to use the full profile of the original curve
                # as the other bound for the new partial face
                for edge in start_edge.loop.edges:
                    new_face_edges.append(edge.geometry)
            else:
                # TODO: the end and start may be reversed for reversed coedges?
                for edge in edges:
                    new_face_edges.append(edge.brep.geometry)
                start_point = None
                start_sketch_point = None
                end_point = None
                end_sketch_point = None
                # TODO: add test where isParamReversed = true
                if start_edge.isParamReversed:
                    start_point = start_edge.edge.endVertex.geometry
                    start_sketch_point = offset_sketch_curves[offset_sketch_curves.count - 1].endSketchPoint.geometry
                else:
                    start_point = start_edge.edge.startVertex.geometry
                    start_sketch_point = offset_sketch_curves[0].startSketchPoint.geometry
                if end_edge.isParamReversed:
                    end_point = end_edge.edge.startVertex.geometry
                    end_sketch_point = offset_sketch_curves[0].startSketchPoint.geometry
                else:
                    end_point = end_edge.edge.endVertex.geometry
                    end_sketch_point = offset_sketch_curves[offset_sketch_curves.count - 1].endSketchPoint.geometry

                new_face_edges.append(
                    sketch.sketchCurves.sketchLines.addByTwoPoints(
                        sketch.modelToSketchSpace(start_point),
                        start_sketch_point).worldGeometry)
                new_face_edges.append(
                    sketch.sketchCurves.sketchLines.addByTwoPoints(
                        sketch.modelToSketchSpace(end_point),
                        end_sketch_point).worldGeometry)

        wire_body, _ = brep().createWireFromCurves(new_face_edges)
        new_face = brep().createFaceFromPlanarWires([wire_body])
        original_face_body = brep().copy(face.brep)

        intersection_body = brep().copy(new_face)

        if loop.isOuter:
            original_face_outer_wires, _ = brep().createWireFromCurves(
                [edge.geometry for edge in _get_outer_loop(face.brep).edges])

            original_face_no_holes = brep().createFaceFromPlanarWires([original_face_outer_wires])
            if not brep().booleanOperation(
                    original_face_no_holes,
                    original_face_body, adsk.fusion.BooleanTypes.DifferenceBooleanType):
                raise ValueError("Couldn't combine the offset parts of the face with the original face")
            only_holes = original_face_no_holes

            # any holes in the original face should remain unfilled after the offset
            if not brep().booleanOperation(new_face, only_holes, adsk.fusion.BooleanTypes.DifferenceBooleanType):
                raise ValueError("Couldn't combine the offset parts of the face with the original face")
        else:
            original_face_outer_wires, _ = brep().createWireFromCurves(
                [edge.geometry for edge in _get_outer_loop(face.brep).edges])
            original_face_outer = brep().createFaceFromPlanarWires([original_face_outer_wires])

            original_hole_wires, _ = brep().createWireFromCurves(
                [edge.geometry for edge in loop.edges])
            original_hole = brep().createFaceFromPlanarWires([original_hole_wires])

            offset_operation_area = original_face_outer
            # The only areas that the offset operation can change are the original hole, or any of the rest of the
            # non-hole area of the face
            if not brep().booleanOperation(offset_operation_area, original_hole, adsk.fusion.BooleanTypes.UnionBooleanType):
                raise ValueError("Couldn't combine the offset parts of the face with the original face")
            if not brep().booleanOperation(new_face, offset_operation_area, adsk.fusion.BooleanTypes.IntersectionBooleanType):
                raise ValueError("Couldn't combine the offset parts of the face with the original face")

        if not brep().booleanOperation(intersection_body, original_face_body, adsk.fusion.BooleanTypes.IntersectionBooleanType):
            raise ValueError("Couldn't combine the offset parts of the face with the original face")

        if not brep().booleanOperation(new_face, original_face_body, adsk.fusion.BooleanTypes.UnionBooleanType):
            raise ValueError("Couldn't combine the offset parts of the face with the original face")

        if not brep().booleanOperation(new_face, intersection_body, adsk.fusion.BooleanTypes.DifferenceBooleanType):
            raise ValueError("Couldn't combine the offset parts of the face with the original face")

        temp_occurrence.deleteMe()

        self._body = new_face
        self._add_children([face.component])
        self._plane = face.brep.geometry

    def _raw_bodies(self) -> Iterable[BRepBody]:
        return self._body,

    def _copy_to(self, copy: 'ComponentWithChildren', copy_children: bool):
        super()._copy_to(copy, copy_children)
        copy._body = self._body
        copy._plane = self._plane

    def get_plane(self) -> Optional[adsk.core.Plane]:
        return self._plane


class SplitFace(ComponentWithChildren):
    """Splits the faces of a Component at the areas where it intersects with another Face or Component.

    Args:
        component: The Component to split the faces of
        splitting_tool: The Face or Component to use to split the faces
        name: The name of the component
    """
    def __init__(self, component: Component, splitting_tool: Onion[Face, Component], name: str = None):
        super().__init__(name)

        temp_occurrence = component.create_occurrence(False)
        faces_to_split = []
        for body in temp_occurrence.bRepBodies:
            faces_to_split.extend(body.faces)

        if isinstance(splitting_tool, Face):
            splitting_component = splitting_tool.component
            body_index, face_index = splitting_component._find_face_index(splitting_tool)
            temp_splitting_occurrence = splitting_component.create_occurrence(False)
            splitting_entities = [temp_splitting_occurrence.bRepBodies[body_index].faces[face_index]]
        elif isinstance(splitting_tool, Component):
            splitting_component = splitting_tool
            temp_splitting_occurrence = splitting_component.create_occurrence(False)
            splitting_entities = list(temp_splitting_occurrence.bRepBodies)
        else:
            raise ValueError("Invalid type for splitting tool: %s" % splitting_tool.__class__.__name__)

        split_face_input = temp_occurrence.component.features.splitFaceFeatures.createInput(
            _collection_of(faces_to_split), _collection_of(splitting_entities), False)
        temp_occurrence.component.features.splitFaceFeatures.add(split_face_input)

        result_faces = []
        for body in temp_occurrence.component.bRepBodies:
            result_faces.extend(body.faces)

        temp_occurrence_bodies = list(temp_occurrence.component.bRepBodies)
        bodies = []
        for body in temp_occurrence_bodies:
            bodies.append(brep().copy(body))
        self._bodies = bodies

        self._split_face_indices = []
        for face in result_faces:
            for splitting_entity in splitting_entities:
                if isinstance(splitting_entity, BRepBody):
                    if _check_face_intersection(face, splitting_entity):
                        body_index = temp_occurrence_bodies.index(face.body)
                        self._split_face_indices.append((body_index, _face_index(face)))
                else:
                    if _check_face_coincidence(face, splitting_entity):
                        body_index = temp_occurrence_bodies.index(face.body)
                        self._split_face_indices.append((body_index, _face_index(face)))

        temp_occurrence.deleteMe()
        temp_splitting_occurrence.deleteMe()

        self._add_children((component, splitting_component))
        self._cached_split_faces = None

    def _copy_to(self, copy: 'ComponentWithChildren', copy_children: bool):
        super()._copy_to(copy, copy_children)
        copy._bodies = list(self._bodies)
        copy._split_face_indices = list(self._split_face_indices)
        copy._cached_split_faces = None

    def _raw_bodies(self) -> Iterable[BRepBody]:
        return self._bodies

    def _reset_cache(self):
        super()._reset_cache()
        self._cached_split_faces = None

    def _get_faces(self, indices) -> Sequence[Face]:
        result = []
        for body_index, face_index in indices:
            result.append(self.bodies[body_index].faces[face_index])
        return result

    @property
    def split_faces(self) -> Sequence[Face]:
        """Returns: The new faces that were created by this Split operation."""
        if not self._cached_split_faces:
            self._cached_split_faces = self._get_faces(self._split_face_indices)
        return list(self._cached_split_faces)


class Silhouette(ComponentWithChildren):
    """Projects the given faces onto a plane.

    Args:
        entity: The faces to revolve. This can be a Face, a Body, a Component, an Edge, or an iterable of Faces or
            Edges.
        plane: The plane to project onto.
        named_edges: An optional mapping of named edges. The edges will be projected onto the silhouette plane, and any
            edges of the Silhouette that match any of the projects edges (similar to find_edges) will be added as a
            named edge.
        name: The name of the component
    """
    def __init__(self, entity: Onion[Component, Body, Edge, Face, Loop, Iterable[Face], Iterable[Edge], Iterable[Loop]],
                 plane: adsk.core.Plane, named_edges: Optional[Mapping[str, Iterable[Edge]]] = None, name: str = None):
        super().__init__(name)

        temp_occurrence = _create_component(root(), name="temp")
        input_component, entities_to_project = self._import_entity_for_projection(temp_occurrence, entity)

        construction_plane_input = temp_occurrence.component.constructionPlanes.createInput(temp_occurrence)
        construction_plane_input.setByPlane(plane)
        construction_plane = temp_occurrence.component.constructionPlanes.add(construction_plane_input)
        sketch = temp_occurrence.component.sketches.add(construction_plane, temp_occurrence)

        silhouette = None
        for entity in entities_to_project:
            projections = sketch.project(entity)

            wires_body = None
            for projection in projections:
                if isinstance(projection.worldGeometry, Point3D):
                    continue
                wire_body, _ = brep().createWireFromCurves((projection.worldGeometry,), False)
                if wires_body is None:
                    wires_body = wire_body
                else:
                    brep().booleanOperation(wires_body, wire_body, adsk.fusion.BooleanTypes.UnionBooleanType)

            try:
                face_silhouette = brep().createFaceFromPlanarWires([wires_body])
            except:
                continue
            if silhouette is None:
                silhouette = face_silhouette
            else:
                brep().booleanOperation(silhouette, face_silhouette, adsk.fusion.BooleanTypes.UnionBooleanType)

        if silhouette is None or silhouette.area == 0:
            silhouette = _create_empty_body()

        self._add_children((input_component,))

        self._body = silhouette
        self._plane = plane

        if named_edges:
            for name, edges in named_edges.items():
                input_component, entities_to_project = self._import_entity_for_projection(temp_occurrence, edges)

                projections = []
                for entity_to_project in entities_to_project:
                    projections.extend(sketch.project(entity_to_project))

                wires_body = None
                for projection in projections:
                    if isinstance(projection.worldGeometry, Point3D):
                        continue
                    wire_body, _ = brep().createWireFromCurves((projection.worldGeometry,), False)
                    if wires_body is None:
                        wires_body = wire_body
                    else:
                        brep().booleanOperation(wires_body, wire_body, adsk.fusion.BooleanTypes.UnionBooleanType)

                wires_body = BRepComponent(wires_body)
                found_edges = self.find_edges(wires_body.edges)
                if found_edges:
                    self.add_named_edges(name, *found_edges)

        temp_occurrence.deleteMe()

    @staticmethod
    def _import_entity_for_projection(
            temp_occurrence: Occurrence,
            entity: Onion[Component, Body, Edge, Face, Loop, Iterable[Face], Iterable[Edge], Iterable[Loop]]):
        input_component = None
        entities_to_project = []
        if isinstance(entity, Component):
            input_component = entity
            for body in entity.bodies:
                entities_to_project.append(temp_occurrence.component.bRepBodies.add(body.brep))
        elif isinstance(entity, Body):
            input_component = entity.component
            entities_to_project.append(temp_occurrence.component.bRepBodies.add(entity.brep))
        elif isinstance(entity, Face):
            input_component = entity.component
            new_body = temp_occurrence.component.bRepBodies.add(brep().copy(entity.brep))
            entities_to_project.append(new_body)
        elif isinstance(entity, Edge):
            input_component = entity.component
            new_body = temp_occurrence.component.bRepBodies.add(brep().copy(entity.brep))
            entities_to_project.append(new_body)
        elif isinstance(entity, Loop):
            input_component = entity.component
            new_body = temp_occurrence.component.bRepBodies.add(brep().copy(entity.brep))
            entities_to_project.append(new_body)
        elif isinstance(entity, Iterable):
            if not entity:
                raise ValueError("No entities were provided")
            all_edges = None
            for subentity in entity:
                if isinstance(subentity, Face):
                    entities_to_project.append(temp_occurrence.component.bRepBodies.add(brep().copy(subentity.brep)))
                elif isinstance(subentity, Edge):
                    if all_edges is None:
                        all_edges = brep().copy(subentity.brep)
                    else:
                        brep().booleanOperation(all_edges,
                                                brep().copy(subentity.brep), adsk.fusion.BooleanTypes.UnionBooleanType)
                else:
                    raise ValueError(
                        "Iterable contains invalid entity type: %s. Expecting Face." % entity.__class__.__name__)
                if input_component is None:
                    input_component = subentity.component
                elif subentity.component != input_component:
                    raise ValueError("All faces and edges must be from the same component")
            if all_edges:
                entities_to_project.append(temp_occurrence.component.bRepBodies.add(all_edges))

        else:
            raise ValueError("Invalid entity type: %s" % entity.__class__.__name__)

        return (input_component, entities_to_project)

    def get_plane(self) -> Optional[adsk.core.Plane]:
        return self._plane

    def _raw_bodies(self) -> Iterable[BRepBody]:
        return self._body,

    def _copy_to(self, copy: 'ComponentWithChildren', copy_children: bool):
        super()._copy_to(copy, copy_children)
        copy._body = self._body
        copy._plane = self._plane


class Hull(ComponentWithChildren):

    def __init__(self, component: Component, tolerance: float = .01, name: str = None):
        super().__init__(name=name)
        plane = component.get_plane()
        if plane is None:
            raise ValueError("Input component must be planar")

        transform = Matrix3D.create()
        transform.setToRotateTo(plane.normal, Vector3D.create(0, 0, 1))

        all_points = []
        for face in component.faces:
            loop_points = []
            for edge in face.outer_edges:
                evaluator = edge.brep.evaluator
                success, start_param, end_param = evaluator.getParameterExtents()
                if not success:
                    raise ValueError("Couldn't get curve extents")
                success, points = evaluator.getStrokes(start_param, end_param, tolerance)
                if not success:
                    raise ValueError("Couldn't get curve strokes")

                if loop_points:
                    if points[0].distanceTo(loop_points[-1]) < app().pointTolerance:
                        points = points[1:]
                    elif points[0].distanceTo(loop_points[0]) < app().pointTolerance:
                        loop_points = loop_points[::-1]
                        points = points[1:]
                    elif points[-1].distanceTo(loop_points[-1]) < app().pointTolerance:
                        points = points[-2::-1]
                    else:
                        loop_points = loop_points[::-1]
                        points = points[-2::-1]
                edge_points = []
                for point in points:
                    point.transformBy(transform)
                    edge_points.append(point)
                loop_points.extend(edge_points)
            assert loop_points[-1].distanceTo(loop_points[0]) < app().pointTolerance
            del(loop_points[-1])
            all_points.extend(loop_points)

        hull_points = Hull._hull(all_points)

        # Note: the last point of the hull points should be a duplicate of the first
        if len(hull_points) < 4:
            raise ValueError("The hull is a line")

        curves = []
        for point1, point2 in zip(hull_points, hull_points[1:]):
            curves.append(adsk.core.Line3D.create(point1, point2))

        wire_body, _ = brep().createWireFromCurves(curves, True)
        hull_body = brep().createFaceFromPlanarWires([wire_body])

        transform.invert()
        brep().transform(hull_body, transform)

        for face in component.faces:
            brep().booleanOperation(hull_body, brep().copy(face.brep), adsk.fusion.BooleanTypes.UnionBooleanType)

        self._body = hull_body

        self._add_children([component])

    def _raw_bodies(self) -> Iterable[BRepBody]:
        return self._body,

    def _copy_to(self, copy: 'ComponentWithChildren', copy_children: bool):
        super()._copy_to(copy, copy_children)
        copy._body = self._body

    def get_plane(self) -> Optional[adsk.core.Plane]:
        if not self.faces:
            return None
        return self.faces[0].get_plane()

    @staticmethod
    def _is_left(p0: Point3D, p1: Point3D, p2: Point3D):
        return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)

    @staticmethod
    def _hull(points: Sequence[Point3D]):
        """

        Derived from http://geomalgorithms.com/a10-_hull-1.html#chainHull_2D()

        :param points:
        :return:
        """

        def point_comparison(p1, p2):
            if p1.x < p2.x:
                return -1
            if p1.x > p2.x:
                return 1
            if p1.y < p2.y:
                return -1
            if p1.y > p2.y:
                return 1
            return 0

        points = sorted(points, key=functools.cmp_to_key(point_comparison))

        def find_last_same_x(_points):

            for i, (point1, point2) in enumerate(zip(_points, _points[1:])):
                if point2.x != point1.x:
                    return i
            return None

        minmin_point = points[0]

        minmax_index = find_last_same_x(points)
        if minmax_index == len(points) - 1:
            return [points[0], points[-1]]
        minmax_point = points[minmax_index]

        maxmin_index = len(points) - 1 - find_last_same_x(points[::-1])
        maxmin_point = points[maxmin_index]

        maxmax_point = points[-1]

        lower_stack = [minmin_point]
        for index in range(minmax_index + 1, maxmin_index + 1):
            point = points[index]
            if Hull._is_left(minmin_point, maxmin_point, point) >= 0 and index < maxmin_index:
                continue
            while len(lower_stack) > 1:
                if Hull._is_left(lower_stack[-2], lower_stack[-1], point) > 0:
                    break
                lower_stack.pop()
            lower_stack.append(point)

        if maxmin_index != len(points) - 1:
            upper_stack = [maxmax_point]
        else:
            upper_stack = [lower_stack.pop()]

        for index in range(maxmin_index - 1, minmax_index - 1, -1):
            point = points[index]

            if Hull._is_left(maxmax_point, minmax_point, point) >= 0 and index > minmax_index:
                continue

            while len(upper_stack) > 1:
                if Hull._is_left(upper_stack[-2], upper_stack[-1], point) > 0:
                    break
                upper_stack.pop()
            upper_stack.append(point)

        if minmax_index != 0:
            upper_stack.append(points[0])

        return lower_stack + upper_stack


class RawThreads(Shape):
    """Creates a raw thread object, not associated with or attached to any cylindrical surface.

    E.g., to create a triangular thread profile with 45 degree upper and lower faces, and a pitch of 1mm::

        threads = RawThreads(
            inner_radius=10,
            pitch=1,
            turns=1,
            thread_profile = [(0, 0), (.5, .5), (0, 1)])

    Args:
        inner_radius: The inner radius of the threads.
        thread_profile: The thread profile as a list of (x, y) tuples. (0, 0) is the "origin" of the thread profile,
            while +x is a vector perpendicular and away from the face of the cylinder, and +y is a vector parallel with
            the axis of the cylinder, pointing toward the top.
        pitch: The pitch of the threads.
        turns: The number of turns of the thread to create.
        name: The name of the component
    """

    def __init__(self, inner_radius: float, thread_profile: Iterable[Tuple[float, float]], pitch: float, turns: float,
                 name: str = None):

        # When the upper point of the thread profile is the same as the lower point of the next turn of the thread,
        # we get an error when creating the ruled surface for the back face about the surface being self-intersecting.
        # In order to avoid this, we split the back profile edge into 2, so that there are 2 separate ruled surfaces
        # that aren't self-intersecting.
        augmented_thread_profile = list(thread_profile)
        augmented_thread_profile.append(((augmented_thread_profile[-1][0] + augmented_thread_profile[0][0]) / 2,
                                         (augmented_thread_profile[-1][1] + augmented_thread_profile[0][1]) / 2))

        start_points = [
            Point3D.create(inner_radius + point[0], 0, point[1])
            for point in augmented_thread_profile]

        helixes = [
            brep().createHelixWire(
                axisPoint=Point3D.create(0, 0, 0),
                axisVector=Vector3D.create(0, 0, 1),
                startPoint=point,
                pitch=pitch,
                turns=turns,
                taperAngle=0) for point in start_points]

        end_points = [helix.edges[0].endVertex.geometry for helix in helixes]

        helix_surfaces = [
            brep().createRuledSurface(first.wires[0], second.wires[0])
            for first, second in _iterate_pairwise(helixes)]

        start_lines = [
            adsk.core.Line3D.create(first, second)
            for first, second in _iterate_pairwise(start_points)]
        start_wires = brep().createWireFromCurves(start_lines, allowSelfIntersections=False)[0]
        start_face_body = brep().createFaceFromPlanarWires([start_wires])

        end_lines = [
            adsk.core.Line3D.create(second, first)
            for first, second in _iterate_pairwise(end_points)]
        end_wires = brep().createWireFromCurves(end_lines, allowSelfIntersections=False)[0]
        end_face_body = brep().createFaceFromPlanarWires([end_wires])

        temp_occurrence = _create_component(
            root(), *helix_surfaces, start_face_body, end_face_body, name="temp")
        stitch_input = temp_occurrence.component.features.stitchFeatures.createInput(
            _collection_of(temp_occurrence.bRepBodies),
            adsk.core.ValueInput.createByReal(app().pointTolerance),
            adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        temp_occurrence.component.features.stitchFeatures.add(stitch_input)

        body = brep().copy(temp_occurrence.bRepBodies[0])
        temp_occurrence.deleteMe()

        planar_face_count = 0
        for index, face in enumerate(body.faces):
            if isinstance(face.geometry, adsk.core.Plane):
                planar_face_count += 1
                to_origin_vector = face.pointOnFace.vectorTo(Point3D.create(0, 0, 0))
                cross = face.geometry.normal.crossProduct(to_origin_vector)
                if cross.z < 0:
                    self._start_face_index = index
                else:
                    self._end_face_index = index

        # the only 2 planar faces should be the start and end faces
        assert(planar_face_count == 2)
        super().__init__(body, name=name)

    @property
    def start_face(self) -> Face:
        """Returns: The flat face at the beginning of the thread. This will be the flat face at the bottom."""
        return self.bodies[0].faces[self._start_face_index]

    @property
    def end_face(self) -> Face:
        """Returns: The flat face at the end of the thread. This will be flat face at the top."""
        return self.bodies[0].faces[self._end_face_index]

    def _copy_to(self, copy: 'Shape', copy_children: bool):
        super()._copy_to(copy, copy_children)
        copy._start_face_index = self._start_face_index
        copy._end_face_index = self._end_face_index


class Threads(ComponentWithChildren):
    """Represents the result of adding threads to a cylindrical object/face.

    E.g., to create a triangular thread profile with 45 degree upper and lower faces, and a pitch of 1mm::

        cylinder = Cylinder(10, 1)
        threaded_cylinder = Threads(cylinder,
                                    [(0, 0), (.5, .5), (0, 1)],
                                    1)

    Args:
        entity: The Component or Face to add threads to. If a Component is given, there must be exactly 1
            cylindrical face present in the Component. If a Face is given, it must be a cylindrical face. In either
            case, a partial cylindrical face is acceptable.
        thread_profile: The thread profile as a list of (x, y) tuples. (0, 0) is the "origin" of the thread profile,
            while +x is a vector perpendicular and away from the face of the cylinder, and +y is a vector parallel with
            the axis of the cylinder, pointing toward the top.
        pitch:
            The distance between each thread
        reverse_axis:
            In case of non-symmetric threads, the direction can be important. Set this to true to reverse the direction
            of the threads, so the top is bottom, and vice versa. Note: This does not change the handed-ness of the
            thread. To make a left-handed thread, you can apply a mirror operation afterward.
        name: The name of the component
    """
    def __init__(self, entity: Onion[Component, Face], thread_profile: Iterable[Tuple[float, float]],
                 pitch: float, reverse_axis=False, name: str = None):
        super().__init__(name)

        if isinstance(entity, Component):
            cylindrical_face = None
            for body in entity.bodies:
                for face in body.faces:
                    if isinstance(face.brep.geometry, adsk.core.Cylinder):
                        if cylindrical_face is None:
                            cylindrical_face = face
                        else:
                            raise ValueError("Found multiple cylindrical faces in component.")
            if cylindrical_face is None:
                raise ValueError("Could not find cylindrical face in component.")
        elif isinstance(entity, Face):
            cylindrical_face = entity
        else:
            raise ValueError("Invalid entity type: %s" % entity.__class__.__name__)

        cylinder = cylindrical_face.brep.geometry
        axis = cylinder.axis
        axis.normalize()

        face_brep = cylindrical_face.brep

        _, thread_start_point = face_brep.evaluator.getPointAtParameter(face_brep.evaluator.parametricRange().minPoint)
        _, end_point = face_brep.evaluator.getPointAtParameter(face_brep.evaluator.parametricRange().maxPoint)
        length = end_point.distanceTo(thread_start_point)

        if reverse_axis:
            origin = cylinder.origin
            axis_temp = axis.copy()
            axis_temp.scaleBy(length)
            origin.translateBy(axis_temp)
            axis_temp = axis.copy()
            axis_temp.scaleBy(-1)
            cylinder = adsk.core.Cylinder.create(origin, axis_temp, cylinder.radius)
            axis = axis_temp
            thread_start_point, end_point = end_point, thread_start_point

        start_point_vector = cylinder.origin.vectorTo(thread_start_point)
        start_point_vector = axis.crossProduct(axis.crossProduct(start_point_vector))
        start_point_vector.scaleBy(-1)
        start_point_vector.normalize()

        start_point_vector_copy = start_point_vector.copy()
        origin = thread_start_point.copy()
        start_point_vector_copy.scaleBy(-1 * cylinder.radius)
        origin.translateBy(start_point_vector_copy)

        max_y = None
        max_x = None
        for point in thread_profile:
            if max_x is None:
                max_x = point[0]
            elif point[0] > max_x:
                max_x = point[0]
            if max_y is None:
                max_y = point[1]
            elif point[1] > max_y:
                max_y = point[1]

        extra_length = math.ceil(max_y / pitch) * pitch
        turns = (length + extra_length)/pitch

        axis_copy = axis.copy()
        axis_copy.scaleBy(-1 * extra_length)
        original_start_point = thread_start_point.copy()
        thread_start_point.translateBy(axis_copy)

        # axis is the "y" axis and start_point_vector is the "x" axis, with thread_start_point as the origin
        helixes = []

        # When the upper point of the thread profile is the same as the lower point of the next turn of the thread,
        # we get an error when creating the ruled surface for the back face about the surface being self-intersecting.
        # In order to avoid this, we split the back profile edge into 2, so that there are 2 separate ruled surfaces
        # that aren't self-intersecting.
        augmented_thread_profile = list(thread_profile)
        augmented_thread_profile.append(((augmented_thread_profile[-1][0] + augmented_thread_profile[0][0]) / 2,
                                         (augmented_thread_profile[-1][1] + augmented_thread_profile[0][1]) / 2))

        for point in augmented_thread_profile:
            start_point = thread_start_point.copy()
            x_axis = start_point_vector.copy()
            x_axis.scaleBy(point[0])
            y_axis = axis.copy()
            y_axis.scaleBy(point[1])
            start_point.translateBy(x_axis)
            start_point.translateBy(y_axis)
            helixes.append(brep().createHelixWire(origin, axis, start_point, pitch, turns, 0))

        face_bodies = []
        start_face_edges = []
        end_face_edges = []
        for i in range(-1, len(helixes)-1):
            face_bodies.append(brep().createRuledSurface(helixes[i].wires[0], helixes[i+1].wires[0]))
            start_face_edges.append(adsk.core.Line3D.create(
                helixes[i].edges[0].startVertex.geometry,
                helixes[i+1].edges[0].startVertex.geometry))
            end_face_edges.append(adsk.core.Line3D.create(
                helixes[i].edges[0].endVertex.geometry,
                helixes[i+1].edges[0].endVertex.geometry))

        start_face_wire, _ = brep().createWireFromCurves(start_face_edges)
        end_face_wire, _ = brep().createWireFromCurves(end_face_edges)

        face_bodies.append(brep().createFaceFromPlanarWires([start_face_wire]))
        face_bodies.append(brep().createFaceFromPlanarWires([end_face_wire]))

        cumulative_body = face_bodies[0]
        for face_body in face_bodies[1:]:
            brep().booleanOperation(cumulative_body, face_body, adsk.fusion.BooleanTypes.UnionBooleanType)

        bottom_plane = adsk.core.Plane.create(original_start_point, axis)
        axis_copy = axis.copy()
        axis_copy.scaleBy(length)
        top_point = original_start_point.copy()
        top_point.translateBy(axis_copy)
        top_plane = adsk.core.Plane.create(top_point, axis)

        surface_bodies = []
        for face in cumulative_body.faces:
            surface_bodies.append(brep().copy(face))

        thread_occurrence = _create_component(root(), *surface_bodies, name="thread")

        stitch_input = thread_occurrence.component.features.stitchFeatures.createInput(
            _collection_of(thread_occurrence.bRepBodies),
            adsk.core.ValueInput.createByReal(app().pointTolerance),
            adsk.fusion.FeatureOperations.NewBodyFeatureOperation)

        thread_occurrence.component.features.stitchFeatures.add(stitch_input)
        cumulative_body = None
        for body in thread_occurrence.bRepBodies:
            if cumulative_body is None:
                cumulative_body = brep().copy(body)
            else:
                brep().booleanOperation(cumulative_body, body, adsk.fusion.BooleanTypes.UnionBooleanType)

        axis_line = adsk.core.InfiniteLine3D.create(cylinder.origin, cylinder.axis)

        point_on_face = face_brep.pointOnFace
        _, normal = face_brep.evaluator.getNormalAtPoint(point_on_face)
        point_projection = _project_point_to_line(point_on_face, axis_line)

        is_male_thread = point_on_face.vectorTo(point_projection).dotProduct(normal) < 0

        if is_male_thread:
            bounding_shell = Thicken(BRepComponent(face_brep), max_x).bodies[0].brep
        else:
            bounding_shell = Thicken(BRepComponent(face_brep), -max_x).bodies[0].brep

        brep().booleanOperation(cumulative_body, bounding_shell, adsk.fusion.BooleanTypes.IntersectionBooleanType)

        base_components = []
        for body in cylindrical_face.component.bodies:
            base_components.append(BRepComponent(body.brep))
        base_component = Union(*base_components)

        thread_component = BRepComponent(cumulative_body)

        if is_male_thread:
            # face normal is outward, and we're adding threads onto the surface
            result = Union(base_component, thread_component)
        else:
            # face normal is inward, and we're cutting threads into the surface
            result = Difference(base_component, thread_component)

        self._bodies = [body.brep for body in result.bodies]
        self._add_children((cylindrical_face.component,))
        thread_occurrence.deleteMe()

    def _raw_bodies(self) -> Iterable[BRepBody]:
        return self._bodies

    def _copy_to(self, copy: 'ComponentWithChildren', copy_children: bool):
        super()._copy_to(copy, copy_children)
        copy._bodies = list(self._bodies)


class Fillet(ComponentWithChildren):
    """Represents a Fillet operation on a set of edges from the same component.

    Args:
        edges: The edges to fillet. All edges must be from the same component.
        radius: The fillet radius
        blend_corners: If true, use fusion's "setback" corner type, otherwise use the "rolling ball" corner type.
        name: The name of the component
    """
    def __init__(self, edges: Iterable[Edge], radius: float, blend_corners: bool = False, name: str = None):
        super().__init__(name)

        component = None
        for edge in edges:
            if component is None:
                component = edge.component
            elif component != edge.component:
                raise ValueError("All edges must be in the same component")

        edge_indices = []
        for edge in edges:
            body_index = _body_index(edge.body, component.bodies)
            if body_index is None:
                raise ValueError("Couldn't find body in component")
            edge_index = _edge_index(edge)
            edge_indices.append((body_index, edge_index))

        occurrence = component.create_occurrence(False)
        occurrence_edges = []
        for edge_index in edge_indices:
            occurrence_edges.append(occurrence.bRepBodies[edge_index[0]].edges[edge_index[1]])

        fillet_input = occurrence.component.features.filletFeatures.createInput()
        fillet_input.addConstantRadiusEdgeSet(_collection_of(occurrence_edges),
                                              ValueInput.createByReal(radius),
                                              False)
        fillet_input.isRollingBallCorner = not blend_corners
        occurrence.component.features.filletFeatures.add(fillet_input)

        self._bodies = [brep().copy(body) for body in occurrence.bRepBodies]

        occurrence.deleteMe()

        self._add_children([component])

    def _raw_bodies(self) -> Iterable[BRepBody]:
        return self._bodies

    def _copy_to(self, copy: 'ComponentWithChildren', copy_children: bool):
        copy._bodies = list(self._bodies)
        super()._copy_to(copy, copy_children)


class Chamfer(ComponentWithChildren):
    """Represents a Chamfer operation on a set of edges from the same component.

    Args:
        edges: The edges to chamfer. All edges must be from the same component.
        distance: The distance from the edge to start the chamfer. If distance2 is not given, this distance is used
            for both sides of the edge.
        distance2: If given, the distance from the other side of the edge to start the chamfer.
        name: The name of the component
    """
    def __init__(self, edges: Iterable[Edge], distance: float, distance2: float = None, name: str = None):
        super().__init__(name)

        component = None
        for edge in edges:
            if component is None:
                component = edge.component
            elif component != edge.component:
                raise ValueError("All edges must be in the same component")

        edge_indices = []
        for edge in edges:
            body_index = _body_index(edge.body, component.bodies)
            if body_index is None:
                raise ValueError("Couldn't find body in component")
            edge_index = _edge_index(edge)
            edge_indices.append((body_index, edge_index))

        occurrence = component.create_occurrence(False)
        occurrence_edges = []
        for edge_index in edge_indices:
            occurrence_edges.append(occurrence.bRepBodies[edge_index[0]].edges[edge_index[1]])

        chamfer_input = occurrence.component.features.chamferFeatures.createInput(
            _collection_of(occurrence_edges), False)
        if distance2 is not None:
            chamfer_input.setToTwoDistances(
                ValueInput.createByReal(distance),
                ValueInput.createByReal(distance2))
        else:
            chamfer_input.setToEqualDistance(
                ValueInput.createByReal(distance))

        feature = occurrence.component.features.chamferFeatures.add(chamfer_input)

        feature_bodies = list(feature.bodies)

        self._bodies = [brep().copy(body) for body in occurrence.bRepBodies]

        self._chamfered_face_indices = []
        for face in feature.faces:
            body_index = feature_bodies.index(face.body)
            self._chamfered_face_indices.append((body_index, _face_index(face)))

        self._cached_chamfered_faces = None

        occurrence.deleteMe()

        self._add_children([component])

    def _raw_bodies(self) -> Iterable[BRepBody]:
        return self._bodies

    def _copy_to(self, copy: 'ComponentWithChildren', copy_children: bool):
        copy._bodies = list(self._bodies)
        copy._chamfered_face_indices = list(self._chamfered_face_indices)
        self._cached_chamfered_faces = None
        super()._copy_to(copy, copy_children)

    def _get_faces(self, indices) -> Sequence[Face]:
        result = []
        for body_index, face_index in indices:
            result.append(self.bodies[body_index].faces[face_index])
        return result

    @property
    def chamfered_faces(self) -> Sequence[Face]:
        """The new chamfered faces that were created."""
        if not self._cached_chamfered_faces:
            self._cached_chamfered_faces = self._get_faces(self._chamfered_face_indices)
        return list(self._cached_chamfered_faces)


class Scale(ComponentWithChildren):
    """Represents a uniform on non-uniform scaling operation on a component

    For uniform scaling, its usually preferred to just use `Component.scale`, which is limited to uniform scaling. This
    class is useful when you need to perform a non-uniform scale.

    Args:
        component: The component to scale
        sx: The scaling ratio in the x axis
        sy: The scaling ratio in the y axis
        sz: The scaling ratio in the z axis
        center: The center of the scaling operation. Defaults to (0, 0, 0) if not specified.
        name: The name of the component
    """
    def __init__(self, component: Component, sx: float = 1, sy: float = 1, sz: float = 1,
                 center: Onion[Point3D, Point, Tuple[float, float, float]] = None, name: str = None):
        super().__init__(name)

        if center is None:
            center = Point3D.create(0, 0, 0)
        elif isinstance(center, Point):
            center = center.point
        elif isinstance(center, Tuple):
            center = Point3D.create(*center)

        occurrence = component.create_occurrence(False)
        try:
            construction_point_input = occurrence.component.constructionPoints.createInput(occurrence)
            construction_point_input.setByPoint(center)
            center_point = occurrence.component.constructionPoints.add(construction_point_input)

            scale_input = occurrence.component.features.scaleFeatures.createInput(_collection_of(occurrence.bRepBodies),
                                                                                  center_point,
                                                                                  ValueInput.createByReal(1))
            scale_input.setToNonUniform(ValueInput.createByReal(sx),
                                        ValueInput.createByReal(sy),
                                        ValueInput.createByReal(sz))
            occurrence.component.features.scaleFeatures.add(scale_input)

            self._bodies = [brep().copy(body) for body in occurrence.bRepBodies]
            self._add_children([component])
        finally:
            occurrence.deleteMe()

    def _raw_bodies(self) -> Iterable[BRepBody]:
        return self._bodies

    def _copy_to(self, copy: 'ComponentWithChildren', copy_children: bool):
        copy._bodies = list(self._bodies)
        super()._copy_to(copy, copy_children)

    def get_plane(self):
        return self.children()[0].get_plane()


class Thicken(ComponentWithChildren):
    """Creates new bodies that consist of the given face(s) made thicker.

    This accepts any number of faces from any number of bodies or components. The resulting bodies will not include the
    bodies that the faces are from.

    A positive thickness
    If thickness is negative, the thickness will be added to the face opposite of its normal, and then unioned with
    the body that the face belongs to, if applicable.

    Args:
        entity: The face to thicken. This can be a Face, Body, Component, or an Iterable of any combination of these.
          For any Body or Component, all faces will be included.
        thickness: How much thickness to add. A positive thickness will add thickness to the front of the face (in
            the direction of the normal), while a negative thickness will add thickness to the back of the face (in the
            direction opposite of the normal)
        name: The name of the component"""

    def __init__(self, entity: _face_selector_types, thickness: float, name: str = None):
        super().__init__(name)

        bodies = []
        body_faces = []
        components = []

        def get_body_index(item):
            try:
                return bodies.index(item)
            except ValueError:
                return -1

        def process_entity(item: _face_selector_types):
            if isinstance(item, Face):
                body_index = get_body_index(item.body)
                if body_index < 0:
                    bodies.append(item.body)
                    body_faces.append([item])
                    if item.component not in components:
                        components.append(item.component)
                else:
                    if item not in body_faces[body_index]:
                        body_faces[body_index].append(item)
            elif isinstance(item, Body):
                body_index = get_body_index(item)
                if body_index < 0:
                    bodies.append(item)
                    body_faces.append(list(item.faces))
                    if item.component not in components:
                        components.append(item.component)
                else:
                    # there's no need to keep track of any faces that were added previously, since we're adding all
                    # faces
                    body_faces[body_index] = list(item.faces)
            elif isinstance(item, Component):
                for subitem in item.bodies:
                    process_entity(subitem)
            elif isinstance(item, Iterable):
                for subitem in item:
                    process_entity(subitem)
            else:
                raise ValueError("Unsupported object type for thicken: %s" % entity.__class__.__name__)

        process_entity(entity)

        temp_occurrence = _create_component(root(), name="temp")

        temp_faces = []
        for i, body in enumerate(bodies):
            if body.brep.isSolid:
                temp_body = temp_occurrence.component.bRepBodies.add(_union_entities(body_faces[i]))
                temp_faces.extend(temp_body.faces)
            else:
                temp_body = temp_occurrence.component.bRepBodies.add(body.brep)
                for face in body_faces[i]:
                    temp_faces.append(_map_face(face, temp_body))

        temp_occurrence.activate()
        result_bodies = []

        thicken_input = temp_occurrence.component.features.thickenFeatures.createInput(
            _collection_of(temp_faces),
            ValueInput.createByReal(thickness),
            False,
            adsk.fusion.FeatureOperations.JoinFeatureOperation,
            False)

        feature = temp_occurrence.component.features.thickenFeatures.add(thicken_input)

        feature_bodies = list(feature.bodies)
        # In some cases, the face being extruded is included in the bodies for some reason. If so, we want to
        # exclude it.
        for i, body in enumerate(feature_bodies):
            if body.isSolid:
                result_bodies.append(BRepComponent(body))

        self._add_children(components)

        self._bodies = [_union_entities(result_bodies)]

        temp_occurrence.deleteMe()

    def _copy_to(self, copy: 'ComponentWithChildren', copy_children: bool):
        super()._copy_to(copy, copy_children)
        copy._bodies = list(self._bodies)

    def _raw_bodies(self) -> Iterable[BRepBody]:
        return self._bodies


# Sequence's __iter__ is implemented via a generator, which doesn't currently place nice with
# fusion 360's SWIG objects (https://forums.autodesk.com/t5/fusion-360-api-and-scripts/strange-stopiteration-issue-caused-by-old-version-of-swig/m-p/8470361#M7106)
class _SequenceIterator(object):
    def __init__(self, sequence):
        self._sequence = sequence
        self._index = 0

    def __iter__(self):
        self._index = 0

    def __next__(self):
        if self._index < len(self._sequence):
            result = self._sequence[self._index]
            self._index += 1
            return result
        raise StopIteration()


class MemoizableDesign(object):
    """
    Serves as a base class for instance-scoped memoization of methods that return a Component

    To use this, create a class that extends from MemoizableDesign, and then you can decorate any member functions
    with @MemoizableDesign.MemoizeComponent. That method will be memoized at the containing instance level, and keyed
    on any arguments passed to the method.

    When the method is called a second time, on the same instance, with the same arguments, a separate copy of the
    memoized component will be returned, using `Component.copy(copy_children=True)`

    If the method has a 'name' argument, it is handled specially. The value of name argument is excluded from the
    memoziation key, but is then passed to `Component.copy()` when making a copy of the memoized value, which ensures
    the name of the returned Component is set to the provided name value.
    """

    def __init__(self):
        self.__memoize_cache = {}

    class MemoizeComponent(object):
        def __init__(self, func):
            self._func = func

        # noinspection PyProtectedMember
        def __call__(self, *args, **kwargs):
            memoizable_instance = args[0]

            try:
                func_cache = memoizable_instance._MemoizableDesign__memoize_cache[self._func]
            except KeyError:
                func_cache = {}
                memoizable_instance._MemoizableDesign__memoize_cache[self._func] = func_cache

            def make_key(val):
                # noinspection PyBroadException
                try:
                    return hash(val)
                except:
                    return id(val)

            sig = inspect.signature(self._func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            key = tuple((item[0], make_key(item[1]))
                   for item in tuple(bound_args.arguments.items())[1:] if item[0] != "name")

            try:
                value = func_cache[key]
                if "name" in bound_args.arguments:
                    return value.copy(copy_children=True, name=bound_args.arguments["name"])
                else:
                    return value.copy(copy_children=True)
            except KeyError:
                value = self._func(*args, **kwargs)
                func_cache[key] = value
                return value.copy(copy_children=True)

        def __get__(self, obj, objtype=None):
            if obj is None:
                return self._func
            return functools.partial(self, obj)


def setup_document(document_name="fSCAD-Preview"):
    """Sets up a fresh document to run a script in.

    This is normally called from run_design instead of being called directly.

    If a document already exists with the given name, it will be forcibly closed (losing any changes, etc.), and
    recreated as an empty document. In addition, the camera position and unit settings from the existing document will
    be saved, and then restored in the new document.

    This enables a script-centric development cycle, where you run the script, view the results in fusion, go back to
    the script to make changes, and re-run the script to recreate the design with the changes you made. In this
    development style, the script is the primary document, while the fusion document is just an ephemeral artifact.

    Args:
        document_name: The name of the document to create. If a document of the given name already exists, it will
            be forcibly closed and recreated.
    """
    preview_doc = None
    saved_camera = None
    saved_units = None
    for document in app().documents:
        if document.name == document_name:
            preview_doc = document
            break
    if preview_doc is not None:
        preview_doc.activate()
        saved_camera = app().activeViewport.camera
        saved_units = design().fusionUnitsManager.distanceDisplayUnits
        preview_doc.close(False)

    preview_doc = app().documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
    preview_doc.name = document_name
    preview_doc.activate()
    if saved_camera is not None:
        is_smooth_transition_bak = saved_camera.isSmoothTransition
        saved_camera.isSmoothTransition = False
        app().activeViewport.camera = saved_camera
        saved_camera.isSmoothTransition = is_smooth_transition_bak
        app().activeViewport.camera = saved_camera
    if saved_units is not None:
        design().fusionUnitsManager.distanceDisplayUnits = saved_units
    design().designType = adsk.fusion.DesignTypes.DirectDesignType


def run_design(design_func, message_box_on_error=True, print_runtime=True, document_name=None,
               design_args=None, design_kwargs=None):
    """Utility method to handle the common setup tasks for a script

    This can be used in a script like this::

        from fscad import *
        def run(_):
            run_design(_design, message_box_on_error=False, document_name=__name__)

    Args:
        design_func: The function that actually creates the design
        message_box_on_error: Set true to pop up a dialog with a stack trace if an error occurs
        print_runtime: If true, print the amount of time the design took to run
        document_name: The name of the document to create. If a document of the given name already exists, it will
            be forcibly closed and recreated.
        design_args: If provided, passed as unpacked position arguments to design_func
        design_kwargs: If provided, passed as unpacked named arguments to design_func
    """
    # noinspection PyBroadException
    try:
        start = time.time()
        if not document_name:
            frame = inspect.stack()[1]
            module = inspect.getmodule(frame[0])
            filename = module.__file__
            document_name = pathlib.Path(filename).stem
        setup_document(document_name)
        design_func(*(design_args or ()), **(design_kwargs or {}))
        end = time.time()
        if print_runtime:
            print("Run time: %f" % (end-start))
    except Exception:
        print(traceback.format_exc())
        if message_box_on_error:
            ui().messageBox('Failed:\n{}'.format(traceback.format_exc()))


def run(_):
    """Entry point for this Fusion 360 plugin.

    This script can be set up to run as a Fusion 360 plugin on startup, so that the fscad module is automatically
    available for use by other scripts.
    """
    fscad = types.ModuleType("fscad.fscad")
    fscad.__path__ = [os.path.dirname(os.path.realpath(__file__))]
    sys.modules['fscad.fscad'] = fscad

    for key in __all__:
        fscad.__setattr__(key, globals()[key])


def relative_import(path):
    """Import a module given a path relative to the calling module.

    This is useful for Fusion 360 scripts, in cases where you need to import a module that's not directly below
    the directory containing the script. The module will be imported as a top level module with a name based on the
    given filename.

    The typical use of this is something like:

        utils = relative_import("../utils.py")

    or

        relative_import("../utils.py")
        from utils import some_utility

    :param path: A relative path to the module to import. e.g. "../utils.py". The path should be relative to the path
        containing the script that is calling the relative_import function.
    :return: The module that was imported.
    """
    caller_path = os.path.abspath(inspect.getfile(inspect.currentframe().f_back))

    script_path = os.path.abspath(os.path.join(os.path.dirname(caller_path), path))
    script_name = os.path.splitext(os.path.basename(script_path))[0]

    sys.path.append(os.path.dirname(script_path))
    try:
        module = importlib.import_module(script_name)
        importlib.reload(module)
        return module
    finally:
        del sys.path[-1]


def stop(_):
    """Callback from Fusion 360 for when this script is being stopped."""
    if "fscad.fscad" in sys.modules:
        del sys.modules["fscad.fscad"]


def _check_all():
    this_module = sys.modules[__name__]

    importable_symbols = []

    for key, value in globals().items():
        # noinspection PyArgumentList
        if not callable(value):
            continue

        if inspect.getmodule(value) != this_module:
            continue

        if key == "run" or key == "stop":
            continue

        if key.startswith("_"):
            continue

        importable_symbols.append(key)

    if importable_symbols != __all__:
        print("__all__ is likely missing some symbols. Expected value:")
        print(importable_symbols)
        raise Exception("__all__ is likely missing some symbols. Expected value: " + str(importable_symbols))

_check_all()
