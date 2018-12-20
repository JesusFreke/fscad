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

from abc import ABC
from adsk.core import BoundingBox3D, Matrix3D, ObjectCollection, OrientedBoundingBox3D, Point3D, ValueInput, Vector3D
from adsk.fusion import BRepBody, BRepFace, BRepFaces, BRepVertex
from typing import Any, Callable, Iterable, Optional, Sequence, Iterator
from typing import Union as Onion  # Haha, why not? Prevents a conflict with our Union type

import adsk.core
import adsk.fusion
import math
import sys
import traceback
import types

# recursive type hints don't actually work yet, so let's expand the recursion a few levels and call it good
_face_selector_types = Onion['Component', 'Body', 'Face', Iterable[
    Onion['Component', 'Body', 'Face', Iterable[
        Onion['Component', 'Body', 'Face', Iterable[
            Onion['Component', 'Body', 'Face', Iterable['_face_selector_types']]]]]]]]


def app():
    return adsk.core.Application.get()


def root() -> adsk.fusion.Component:
    return design().rootComponent


def ui():
    return app().userInterface


def brep():
    return adsk.fusion.TemporaryBRepManager.get()


def design():
    return adsk.fusion.Design.cast(app().activeProduct)


def _is_parametric():
    return design().designType == adsk.fusion.DesignTypes.ParametricDesignType


def set_parametric(parametric):
    if parametric:
        design().designType = adsk.fusion.DesignTypes.ParametricDesignType
    else:
        design().designType = adsk.fusion.DesignTypes.DirectDesignType


def _collection_of(collection):
    object_collection = ObjectCollection.create()
    for obj in collection:
        object_collection.add(obj)
    return object_collection


def _create_component(parent_component, *bodies: Onion[BRepBody, 'Body'], name):
    parametric = _is_parametric()
    new_occurrence = parent_component.occurrences.addNewComponent(Matrix3D.create())
    new_occurrence.component.name = name
    base_feature = None
    if parametric:
        base_feature = new_occurrence.component.features.baseFeatures.add()
        base_feature.startEdit()
    for body in bodies:
        if isinstance(body, Body):
            body = body.brep
        new_occurrence.component.bRepBodies.add(body, base_feature)
    if base_feature:
        base_feature.finishEdit()
    return new_occurrence


def _oriented_bounding_box_to_bounding_box(oriented: OrientedBoundingBox3D):
    return BoundingBox3D.create(
        Point3D.create(
            oriented.centerPoint.x - oriented.length / 2.0,
            oriented.centerPoint.y - oriented.width / 2.0,
            oriented.centerPoint.z - oriented.height / 2.0),
        Point3D.create(
            oriented.centerPoint.x + oriented.length / 2.0,
            oriented.centerPoint.y + oriented.width / 2.0,
            oriented.centerPoint.z + oriented.height / 2.0)
    )


def _get_exact_bounding_box(entity):
    vector1 = adsk.core.Vector3D.create(1.0, 0.0, 0.0)
    vector2 = adsk.core.Vector3D.create(0.0, 1.0, 0.0)

    if isinstance(entity, Component):
        entities = entity.bodies()
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


def _face_index(face: Onion[BRepFace, 'Face']):
    if isinstance(face, Face):
        return _face_index(face.brep)
    for i, candidate_face in enumerate(face.body.faces):
        if candidate_face == face:
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
        return _flatten_face_selectors(selector.bodies())
    if isinstance(selector, Body):
        return selector.brep.faces
    return selector.brep,


def _check_face_intersection(face1, face2):
    face_body1 = brep().copy(face1)
    face_body2 = brep().copy(face2)
    brep().booleanOperation(face_body1, face_body2, adsk.fusion.BooleanTypes.IntersectionBooleanType)
    return face_body1.faces.count > 0


def _point_vector_to_line(point, vector):
    return adsk.core.Line3D.create(point,
                                   adsk.core.Point3D.create(point.x + vector.x,
                                                            point.y + vector.y,
                                                            point.z + vector.z))


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


def _find_coincident_faces_on_body(body: BRepBody, faces: Iterable[BRepFace]) -> Iterable[BRepFace]:
    coincident_faces = []
    for face in faces:
        face_bounding_box = face.boundingBox
        expanded_bounding_box = adsk.core.BoundingBox3D.create(
            adsk.core.Point3D.create(
                face_bounding_box.minPoint.x - app().pointTolerance,
                face_bounding_box.minPoint.y - app().pointTolerance,
                face_bounding_box.minPoint.z - app().pointTolerance),
            adsk.core.Point3D.create(
                face_bounding_box.maxPoint.x + app().pointTolerance,
                face_bounding_box.maxPoint.y + app().pointTolerance,
                face_bounding_box.maxPoint.z + app().pointTolerance),
        )
        if body.boundingBox.intersects(expanded_bounding_box):
            for body_face in body.faces:
                if body_face.boundingBox.intersects(expanded_bounding_box):
                    if _check_face_coincidence(face, body_face):
                        coincident_faces.append(body_face)
    return coincident_faces


class Translation(object):
    def __init__(self, vector: Vector3D):
        self._vector = vector

    def vector(self):
        return self.vector

    def __add__(self, other):
        self._vector.setWithArray((self._vector.x + other, self._vector.y + other, self._vector.z + other))
        return self

    def __sub__(self, other):
        self._vector.setWithArray((self._vector.x - other, self._vector.y - other, self._vector.z - other))
        return self

    def __mul__(self, other):
        self._vector.setWithArray((self._vector.x * other, self._vector.y * other, self._vector.z * other))
        return self

    def __div__(self, other):
        self._vector.setWithArray((self._vector.x / other, self._vector.y / other, self._vector.z / other))
        return self

    @property
    def x(self):
        return self._vector.x

    @property
    def y(self):
        return self._vector.y

    @property
    def z(self):
        return self._vector.z


class Place(object):
    def __init__(self, point: Point3D):
        self._point = point

    def __eq__(self, other: Onion['Place', float, int, Point3D]) -> Translation:
        if isinstance(other, Point3D):
            point = other
        elif isinstance(other, float) or isinstance(other, int):
            point = Point3D.create(other, other, other)
        elif isinstance(other, Place):
            point = other._point
        else:
            raise ValueError("Unsupported type: %s" % type(other).__name__)

        return Translation(self._point.vectorTo(point))


class BoundedEntity(object):
    def __init__(self):
        self._cached_bounding_box = None

    @property
    def bounding_box(self) -> BoundingBox3D:
        if self._cached_bounding_box is None:
            self._cached_bounding_box = self._calculate_bounding_box()
        return self._cached_bounding_box

    def _calculate_bounding_box(self) -> BoundingBox3D:
        raise NotImplementedError()

    def _reset_cache(self):
        self._cached_bounding_box = None

    def size(self):
        return self.bounding_box.minPoint.vectorTo(self._cached_bounding_box.maxPoint).asPoint()

    def min(self):
        return self.bounding_box.minPoint

    def max(self):
        return self.bounding_box.maxPoint

    def mid(self):
        return Point3D.create(
            (self.bounding_box.minPoint.x + self.bounding_box.maxPoint.x)/2,
            (self.bounding_box.minPoint.y + self.bounding_box.maxPoint.y)/2,
            (self.bounding_box.minPoint.z + self.bounding_box.maxPoint.z)/2)

    def __neg__(self):
        return Place(self.min())

    def __pos__(self):
        return Place(self.max())

    def __invert__(self):
        return Place(self.mid())


class BRepEntity(BoundedEntity, ABC):
    def __init__(self, component: 'Component'):
        super().__init__()
        self._component = component

    @property
    def brep(self) -> Any:  # TODO: define union of all possible brep types?
        raise NotImplementedError

    @property
    def component(self) -> 'Component':
        return self._component

    def _calculate_bounding_box(self) -> BoundingBox3D:
        return _get_exact_bounding_box(self.brep)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return brep == other.brep

    def __ne__(self, other):
        if not isinstance(other, self.__class__):
            return True
        return brep != other.brep


class Body(BRepEntity):
    def __init__(self, body: BRepBody, component: 'Component'):
        super().__init__(component)
        self._body = body

    @property
    def brep(self) -> BRepBody:
        return self._body

    @property
    def faces(self) -> Sequence['Face']:
        class Faces(Sequence['Face']):
            def __init__(self, faces: Onion[Sequence[BRepFace], BRepFaces], body: 'Body'):
                self._faces = faces
                self._body = body

            def __getitem__(self, i: Onion[int, slice]) -> Onion['Face', Sequence['Face']]:
                if isinstance(i, slice):
                    return Faces(self._faces[i], self._body)
                return Face(self._faces[i], self._body)

            def __iter__(self) -> Iterator['Face']:
                # Sequence's __iter__ is implemented via a generator, which doesn't currently place nice with
                # fusion 360's SWIG objects
                class FaceIter(object):
                    def __init__(self, faces):
                        self._faces = faces
                        self._index = 0

                    def __iter__(self):
                        self._index = 0

                    def __next__(self):
                        if self._index < len(self._faces):
                            result = self._faces[self._index]
                            self._index += 1
                            return result
                        raise StopIteration()
                return FaceIter(self)

            def __len__(self) -> int:
                return len(self._faces)
        return Faces(self._body.faces, self)


class Face(BRepEntity):
    def __init__(self, face: BRepFace, body: Body):
        super().__init__(body.component)
        self._face = face
        self._body = body

    @property
    def brep(self) -> BRepFace:
        return self._face

    @property
    def body(self) -> Body:
        return self._body


class Component(BoundedEntity):
    _origin = Point3D.create(0, 0, 0)
    _null_vector = Vector3D.create(0, 0, 0)
    _pos_x = Vector3D.create(1, 0, 0)
    _pos_y = Vector3D.create(0, 1, 0)
    _pos_z = Vector3D.create(0, 0, 1)

    name = ...  # type: Optional[str]

    def __init__(self, name: str = None):
        super().__init__()
        self._parent = None
        self._local_transform = Matrix3D.create()
        self.name = name
        self._cached_bodies = None
        self._cached_world_transform = None
        self._cached_inverse_transform = None
        self._named_points = {}

    def _calculate_bounding_box(self) -> BoundingBox3D:
        return _get_exact_bounding_box(self.bodies())

    def _raw_bodies(self) -> Iterable[BRepBody]:
        raise NotImplementedError()

    def copy(self) -> 'Component':
        copy = Component()
        copy.__class__ = self.__class__
        copy._local_transform = self._get_world_transform()
        copy._cached_bounding_box = None
        copy._cached_bodies = None
        copy._cached_world_transform = None
        copy._cached_inverse_transform = None
        copy.name = self.name
        self._copy_to(copy)
        return copy

    def _copy_to(self, copy: 'Component'):
        raise NotImplementedError

    def children(self) -> Iterable['Component']:
        return ()

    def _default_name(self) -> str:
        return self.__class__.__name__

    @property
    def parent(self) -> 'Component':
        return self._parent

    def bodies(self) -> Sequence[Body]:
        if self._cached_bodies is not None:
            return self._cached_bodies

        world_transform = self._get_world_transform()
        bodies_copy = [Body(brep().copy(body), self) for body in self._raw_bodies()]
        for body in bodies_copy:
            brep().transform(body.brep, world_transform)
        self._cached_bodies = tuple(bodies_copy)
        return bodies_copy

    def create_occurrence(self, create_children=False) -> adsk.fusion.Occurrence:
        occurrence = _create_component(root(), *self.bodies(), name=self.name or self._default_name())
        if create_children:
            for child in self.children():
                child._create_occurrence(occurrence)
        return occurrence

    def _create_occurrence(self, parent_occurrence):
        occurrence = _create_component(
            parent_occurrence.component, *self.bodies(), name=self.name or self._default_name())
        occurrence.isLightBulbOn = False
        for child in self.children():
            child._create_occurrence(occurrence)

    def place(self, x=_null_vector, y=_null_vector, z=_null_vector):
        transform = Matrix3D.create()
        transform.translation = Vector3D.create(x.x, y.y, z.z)
        self._local_transform.transformBy(transform)
        self._reset_cache()
        return self

    def _reset_cache(self):
        super()._reset_cache()
        self._cached_bodies = None
        self._cached_world_transform = None
        self._cached_inverse_transform = None
        for component in self.children():
            component._reset_cache()

    def _get_world_transform(self) -> Matrix3D:
        if self._cached_world_transform is not None:
            return self._cached_world_transform.copy()
        transform = self._local_transform.copy()
        if self.parent is not None:
            transform.transformBy(self.parent._get_world_transform())
        self._cached_world_transform = transform
        return transform.copy()

    def _inverse_world_transform(self):
        if self._cached_inverse_transform is None:
            self._cached_inverse_transform = self._get_world_transform()
            self._cached_inverse_transform.invert()
        return self._cached_inverse_transform

    def get_plane(self) -> Optional[adsk.core.Plane]:
        return None

    def rotate(self, rx: float = 0, ry: float = 0, rz: float = 0,
               center: Onion[Iterable[Onion[float, int]], Point3D]=None) -> 'Component':
        transform = self._local_transform

        if center is None:
            center_point = self._origin
        elif isinstance(center, Point3D):
            center_point = center
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

    def rx(self, angle: float, center: Onion[Iterable[Onion[float, int]], Point3D]=None) -> 'Component':
        return self.rotate(angle, center=center)

    def ry(self, angle: float, center: Onion[Iterable[Onion[float, int]], Point3D]=None) -> 'Component':
        return self.rotate(ry=angle, center=center)

    def rz(self, angle: float, center: Onion[Iterable[Onion[float, int]], Point3D]=None) -> 'Component':
        return self.rotate(rz=angle, center=center)

    def translate(self, tx: float = 0, ty: float = 0, tz: float = 0) -> 'Component':
        translation = Matrix3D.create()
        translation.translation = adsk.core.Vector3D.create(tx, ty, tz)
        self._local_transform.transformBy(translation)
        self._reset_cache()
        return self

    def tx(self, tx: float) -> 'Component':
        return self.translate(tx)

    def ty(self, ty: float) -> 'Component':
        return self.translate(ty=ty)

    def tz(self, tz: float) -> 'Component':
        return self.translate(tz=tz)

    def scale(self, sx: float = 1, sy: float = 1, sz: float = 1,
              center: Onion[Iterable[Onion[float, int]], Point3D]=None) -> 'Component':
        scale = Matrix3D.create()
        translation = Matrix3D.create()
        if abs(sx) != abs(sy) or abs(sy) != abs(sz):
            raise ValueError("Non-uniform scaling is not currently supported")

        if center is None:
            center_point = self._origin
        elif isinstance(center, Point3D):
            center_point = center
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

    def find_faces(self, selector: _face_selector_types) -> Sequence[Face]:
        selector_faces = _flatten_face_selectors(selector)
        result = []
        for body in self.bodies():
            for face in _find_coincident_faces_on_body(body.brep, selector_faces):
                result.append(Face(face, body))
        return result

    def add_point(self, name: str, point: Onion[Sequence[float], Point3D]):
        if not isinstance(point, Point3D):
            point = Point3D.create(*point)
        else:
            point = point.copy()
        point.transformBy(self._inverse_world_transform())
        self._named_points[name] = point

    def point(self, name) -> Optional[Point3D]:
        point = self._named_points.get(name)
        if not point:
            return None
        point = point.copy()
        point.transformBy(self._get_world_transform())
        return point


class Shape(Component, ABC):
    def __init__(self, body: BRepBody, name: str):
        super().__init__(name)
        self._body = body

    def _raw_bodies(self):
        return self._body,

    def _copy_to(self, copy: 'Shape'):
        copy._body = brep().copy(self._body)


class PlanarShape(Shape):
    def __init__(self, body: BRepBody, name: str):
        super().__init__(body, name)

    def get_plane(self) -> adsk.core.Plane:
        return self.bodies()[0].faces[0].brep.geometry


class Box(Shape):
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
        super().__init__(body, name)

    @property
    def top(self):
        return self.bodies()[0].faces[self._top_index]

    @property
    def bottom(self):
        return self.bodies()[0].faces[self._bottom_index]

    @property
    def left(self):
        return self.bodies()[0].faces[self._left_index]

    @property
    def right(self):
        return self.bodies()[0].faces[self._right_index]

    @property
    def front(self):
        return self.bodies()[0].faces[self._front_index]

    @property
    def back(self):
        return self.bodies()[0].faces[self._back_index]


class Cylinder(Shape):
    _side_index = 0

    def __init__(self, height: float, radius: float, top_radius: float = None, name: str = None):
        if radius == 0:
            # The API doesn't support the bottom radius being 0, so create it in the opposite orientation and flip it
            body = brep().createCylinderOrCone(self._origin, top_radius, Point3D.create(0, 0, height), radius)
            # 180 degrees around the x axis
            rotation = Matrix3D.create()
            rotation.setCell(1, 1, -1)
            rotation.setCell(2, 2, -1)
            translation = Matrix3D.create()
            translation.translation = Vector3D.create(0, 0, height)
            rotation.transformBy(translation)
            brep().transform(body, rotation)
        else:
            body = brep().createCylinderOrCone(self._origin, radius, Point3D.create(0, 0, height),
                                               top_radius if top_radius is not None else radius)
        if radius == 0:
            self._bottom_index = None
            self._top_index = 1
        elif top_radius == 0:
            self._bottom_index = 1
            self._top_index = None
        else:
            self._bottom_index = 1
            self._top_index = 2

        super().__init__(body, name)

    def _copy_to(self, copy: 'Cylinder'):
        super()._copy_to(copy)
        copy._bottom_index = self._bottom_index
        copy._top_index = self._top_index

    @property
    def top(self):
        if self._top_index is None:
            return None
        return self.bodies()[0].faces[self._top_index]

    @property
    def bottom(self):
        if self._bottom_index is None:
            return None
        return self.bodies()[0].faces[self._bottom_index]

    @property
    def side(self):
        return self.bodies()[0].faces[self._side_index]


class Sphere(Shape):
    def __init__(self, radius: float, name: str = None):
        super().__init__(brep().createSphere(self._origin, radius), name)

    @property
    def surface(self):
        return self.bodies()[0].faces[0]


class Rect(PlanarShape):
    def __init__(self, x: float, y: float, name: str = None):
        # this is a bit faster than creating it from createWireFromCurves -> createFaceFromPlanarWires
        box = brep().createBox(OrientedBoundingBox3D.create(
            Point3D.create(x/2, y/2, -.5),
            self._pos_x, self._pos_y,
            x, y, 1))
        super().__init__(brep().copy(box.faces[Box._top_index]), name)


class Circle(PlanarShape):
    _top_index = 2

    def __init__(self, radius: float, name: str = None):
        # this is a bit faster than creating it from createWireFromCurves -> createFaceFromPlanarWires
        cylinder = brep().createCylinderOrCone(
            Point3D.create(0, 0, -1), radius, self._origin, radius)
        super().__init__(brep().copy(cylinder.faces[self._top_index]), name)


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

    def children(self) -> Iterable['Component']:
        return tuple(self._children)

    def _copy_to(self, copy: 'ComponentWithChildren'):
        copy._cached_inverse_transform = None
        copy._children = []
        copy._add_children([child.copy() for child in self._children])


class Union(ComponentWithChildren):
    def __init__(self, *components: Component, name: str = None):
        super().__init__(name)
        self._body = None

        def process_child(child: Component):
            self._check_coplanarity(child)
            for body in child.bodies():
                if self._body is None:
                    self._body = brep().copy(body.brep)
                    self._plane = child.get_plane()
                else:
                    brep().booleanOperation(self._body, body.brep, adsk.fusion.BooleanTypes.UnionBooleanType)
        self._add_children(components, process_child)

    def _raw_bodies(self) -> Iterable[BRepBody]:
        return self._body,

    def _copy_to(self, copy: 'Union'):
        copy._body = brep().copy(self._body)
        super()._copy_to(copy)

    def _check_coplanarity(self, child):
        if self._body is not None:
            plane = self.get_plane()
            child_plane = child.get_plane()

            if (child_plane is None) ^ (plane is None):
                raise ValueError("Cannot union a planar entity with a 3d entity")
            if plane is not None and not plane.isCoPlanarTo(child_plane):
                raise ValueError("Cannot union planar entities that are non-coplanar")

    def add(self, *components: Component) -> Component:
        def process_child(child):
            self._check_coplanarity(child)
            for body in child.bodies():
                brep().booleanOperation(self._body, body.brep, adsk.fusion.BooleanTypes.UnionBooleanType)
        self._add_children(components, process_child)
        return self

    def _first_child(self):
        try:
            return next(iter(self.children()))
        except StopIteration:
            return None

    def get_plane(self) -> Optional[adsk.core.Plane]:
        child = self._first_child()
        return child.get_plane() if child is not None else None


class Difference(ComponentWithChildren):
    def __init__(self, *components: Component, name: str = None):
        super().__init__(name)
        self._bodies = None

        def process_child(child: Component):
            self._check_coplanarity(child)
            if self._bodies is None:
                self._bodies = [brep().copy(child_body.brep) for child_body in child.bodies()]
            else:
                for target_body in self._bodies:
                    for tool_body in child.bodies():
                        brep().booleanOperation(target_body, tool_body.brep,
                                                adsk.fusion.BooleanTypes.DifferenceBooleanType)
        self._add_children(components, process_child)

    def _raw_bodies(self) -> Iterable[BRepBody]:
        return self._bodies

    def _copy_to(self, copy: 'Difference'):
        copy._bodies = [brep().copy(body) for body in self.bodies()]
        super()._copy_to(copy)

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

    def add(self, *components: Component) -> Component:
        def process_child(child):
            self._check_coplanarity(child)
            for target_body in self._bodies:
                for tool_body in child.bodies():
                    brep().booleanOperation(target_body, tool_body.brep, adsk.fusion.BooleanTypes.DifferenceBooleanType)
        self._add_children(components, process_child)
        return self

    def _first_child(self):
        try:
            return next(iter(self.children()))
        except StopIteration:
            return None

    def get_plane(self) -> Optional[adsk.core.Plane]:
        child = self._first_child()
        return child.get_plane() if child is not None else None


class Intersection(ComponentWithChildren):
    def __init__(self, *components: Component, name: str = None):
        super().__init__(name)
        self._bodies = None
        self._cached_plane = None
        self._cached_plane_populated = False

        plane = None

        def process_child(child: Component):
            nonlocal plane
            plane = self._check_coplanarity(child, plane)
            if self._bodies is None:
                self._bodies = [brep().copy(child_body.brep) for child_body in child.bodies()]
            else:
                for target_body in self._bodies:
                    for tool_body in child.bodies():
                        brep().booleanOperation(target_body, tool_body.brep,
                                                adsk.fusion.BooleanTypes.IntersectionBooleanType)
        self._add_children(components, process_child)

    def _raw_bodies(self) -> Iterable[BRepBody]:
        return self._bodies

    def _copy_to(self, copy: 'Difference'):
        copy._bodies = [brep().copy(body) for body in self.bodies()]
        copy._cached_plane = None
        copy._cached_plane_populated = False
        super()._copy_to(copy)

    def add(self, *components: Component) -> Component:
        plane = self.get_plane()

        def process_child(child):
            nonlocal plane
            plane = self._check_coplanarity(child, plane)
            for target_body in self._bodies:
                for tool_body in child.bodies():
                    brep().booleanOperation(target_body, tool_body.brep,
                                            adsk.fusion.BooleanTypes.IntersectionBooleanType)
        self._add_children(components, process_child)
        return self

    def get_plane(self) -> Optional[adsk.core.Plane]:
        if not self._cached_plane_populated:
            plane = None
            for child in self.children():
                plane = child.get_plane()
                if plane:
                    break
            self._cached_plane = plane
            self._cached_plane_populated = True
        return self._cached_plane

    def _check_coplanarity(self, child, plane):
        if self._bodies is not None and len(self._bodies) > 0:
            child_plane = child.get_plane()
            if plane is not None:
                if child_plane is not None and not plane.isCoPlanarTo(child_plane):
                    raise ValueError("Cannot intersect planar entities that are non-coplanar")
                return plane
            elif child_plane is not None:
                return child_plane
        else:
            return child.get_plane()

    def _reset_cache(self):
        super()._reset_cache()
        self._cached_plane = None
        self._cached_plane_populated = False


class Loft(ComponentWithChildren):
    _top_index = 0
    _bottom_index = 1

    def __init__(self, *components: Component, name: str = None):
        super().__init__(name)

        loft_sections = []

        def process_child(child: Component):
            nonlocal loft_sections
            if child.get_plane() is None:
                raise ValueError("Only planar geometry can be used with Loft")

            component_face = None
            for child_body in child.bodies():
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
        # TODO: do we get a feature in direct mode? If so, we can use as a better way to find the various bodies/faces
        occurrence.component.features.loftFeatures.add(loft_feature_input)
        self._body = brep().copy(occurrence.bRepBodies[-1])
        occurrence.deleteMe()

    def _raw_bodies(self) -> Iterable[BRepBody]:
        return self._body,

    def _copy_to(self, copy: 'Loft'):
        copy._body = brep().copy(self._body)
        super()._copy_to(copy)

    @property
    def bottom(self) -> Face:
        return self.bodies()[0].faces[self._bottom_index]

    @property
    def top(self) -> Face:
        return self.bodies()[0].faces[self._top_index]

    @property
    def sides(self) -> Iterable[Face]:
        return tuple(self.bodies()[0].faces[self._bottom_index+1:])


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
        feature = temp_occurrence.component.features.extrudeFeatures.add(extrude_input)

        bodies = []
        feature_bodies = list(feature.bodies)
        for body in feature_bodies:
            bodies.append(brep().copy(body))
        self._bodies = bodies

        self._start_face_indices = []
        for face in feature.startFaces:
            body_index = feature_bodies.index(face.body)
            self._start_face_indices.append((body_index, _face_index(face)))

        self._end_face_indices = []
        for face in feature.endFaces:
            body_index = feature_bodies.index(face.body)
            self._end_face_indices.append((body_index, _face_index(face)))

        self._side_face_indices = []
        for face in feature.sideFaces:
            body_index = feature_bodies.index(face.body)
            self._side_face_indices.append((body_index, _face_index(face)))

        self._add_children([component])

        self._cached_start_faces = None
        self._cached_end_faces = None
        self._cached_side_faces = None

        temp_occurrence.deleteMe()

    def _copy_to(self, copy: 'Component'):
        copy._bodies = list(self._bodies)
        copy._start_face_indices = list(self._start_face_indices)
        copy._end_face_indices = list(self._end_face_indices)
        copy._side_face_indices = list(self._side_face_indices)

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
            result.append(self.bodies()[body_index].faces[face_index])
        return result

    @property
    def start_faces(self) -> Sequence[Face]:
        if not self._cached_start_faces:
            self._cached_start_faces = self._get_faces(self._start_face_indices)
        return list(self._cached_start_faces)

    @property
    def end_faces(self) -> Sequence[Face]:
        if not self._cached_end_faces:
            self._cached_end_faces = self._get_faces(self._end_face_indices)
        return list(self._cached_end_faces)

    @property
    def side_faces(self) -> Sequence[Face]:
        if not self._cached_side_faces:
            self._cached_side_faces = self._get_faces(self._side_face_indices)
        return list(self._cached_side_faces)


class Extrude(ExtrudeBase):
    def __init__(self, component: Component, height: float, name: str = None):
        if component.get_plane() is None:
            raise ValueError("Can't extrude non-planar geometry with Extrude. Consider using ExtrudeFace")
        faces = []
        for body in component.bodies():
            faces.extend(body.brep.faces)
        super().__init__(
            component, faces, adsk.fusion.DistanceExtentDefinition.create(adsk.core.ValueInput.createByReal(height)),
            name)


class ExtrudeTo(ExtrudeBase):
    def __init__(self, component: Component,
                 to_entity: Onion[Component, Face, Body],
                 name: str = None):
        if component.get_plane() is None:
            raise ValueError("Can't extrude non-planar geometry with Extrude. Consider using ExtrudeFace")
        faces = []
        for body in component.bodies():
            faces.extend(body.brep.faces)
        if isinstance(to_entity, Component):
            bodies = to_entity.bodies()
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

        super().__init__(component, faces, adsk.fusion.ToEntityExtentDefinition.create(to_entity, False), name)
        self._add_children([component_to_add])
        temp_occurrence.deleteMe()


def setup_document(document_name="fSCAD-Preview"):
    preview_doc = None
    saved_camera = None
    for document in app().documents:
        if document.name == document_name:
            preview_doc = document
            break
    if preview_doc is not None:
        preview_doc.activate()
        saved_camera = app().activeViewport.camera
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


def run_design(design_func, message_box_on_error=True, document_name="fSCAD-Preview"):
    """
    Utility method to handle the common setup tasks for a script

    :param design_func: The function that actually creates the design
    :param message_box_on_error: Set true to pop up a dialog with a stack trace if an error occurs
    :param document_name: The name of the document to create. If a document of the given name already exists, it will
    be forcibly closed and recreated.
    """
    # noinspection PyBroadException
    try:
        setup_document(document_name)
        design_func()
    except Exception:
        print(traceback.format_exc())
        if message_box_on_error:
            ui().messageBox('Failed:\n{}'.format(traceback.format_exc()))


def run(_):
    fscad = types.ModuleType("fscad")
    sys.modules['fscad'] = fscad

    for key, value in globals().items():
        # noinspection PyArgumentList
        if not callable(value):
            continue
        if key == "run" or key == "stop":
            continue
        fscad.__setattr__(key, value)


def stop(_):
    del sys.modules['fscad']
