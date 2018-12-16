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

from adsk.core import BoundingBox3D, Matrix3D, ObjectCollection, OrientedBoundingBox3D, Point3D, Vector3D
from typing import Iterable, Optional

import adsk.core
import adsk.fusion
import sys
import traceback
import types


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


def _create_component(parent_component, *bodies, name):
    parametric = _is_parametric()
    if not parametric:
        set_parametric(True)
    new_occurrence = parent_component.occurrences.addNewComponent(Matrix3D.create())
    new_occurrence.component.name = name
    base_feature = new_occurrence.component.features.baseFeatures.add()
    base_feature.startEdit()
    for body in bodies:
        new_occurrence.component.bRepBodies.add(body, base_feature)
    base_feature.finishEdit()
    if not parametric:
        set_parametric(False)
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

    def __mul__(self, other):
        self._vector.setWithArray((self._vector.x * other, self._vector.y * other, self._vector.z * other))

    def __div__(self, other):
        self._vector.setWithArray((self._vector.x / other, self._vector.y / other, self._vector.z / other))

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

    def __eq__(self, other: 'Place') -> Translation:
        return Translation(self._point.vectorTo(other._point))


class Component(object):
    _null_vector = Vector3D.create(0, 0, 0)

    name = ...  # type: Optional[str]

    def __init__(self, name: str = None):
        self._parent = None
        self._local_transform = Matrix3D.create()
        self.name = name
        self._cached_bounding_box = None
        self._cached_bodies = None
        self._cached_world_transform = None

    def _raw_bodies(self) -> Iterable[adsk.fusion.BRepBody]:
        raise NotImplementedError()

    def children(self) -> Iterable['Component']:
        return ()

    def _default_name(self) -> str:
        return "Component"

    @property
    def parent(self) -> 'Component':
        return self._parent

    def bodies(self) -> Iterable[adsk.fusion.BRepBody]:
        if self._cached_bodies is not None:
            return self._cached_bodies

        world_transform = self._get_world_transform()
        bodies_copy = [brep().copy(body) for body in self._raw_bodies()]
        for body in bodies_copy:
            brep().transform(body, world_transform)
        self._cached_bodies = bodies_copy
        return bodies_copy

    def create_occurrence(self) -> adsk.fusion.Occurrence:
        return _create_component(root(), *self.bodies(), name=self.name or self._default_name())

    def size(self):
        if not self._cached_bounding_box:
            self._cached_bounding_box = _get_exact_bounding_box(self)
        return self._cached_bounding_box.minPoint.vectorTo(self._cached_bounding_box.maxPoint).asPoint()

    def min(self):
        if not self._cached_bounding_box:
            self._cached_bounding_box = _get_exact_bounding_box(self)
        return self._cached_bounding_box.minPoint

    def max(self):
        if not self._cached_bounding_box:
            self._cached_bounding_box = _get_exact_bounding_box(self)
        return self._cached_bounding_box.maxPoint

    def mid(self):
        if not self._cached_bounding_box:
            self._cached_bounding_box = _get_exact_bounding_box(self)
        return Point3D.create(
            (self._cached_bounding_box.minPoint.x + self._cached_bounding_box.maxPoint.x)/2,
            (self._cached_bounding_box.minPoint.y + self._cached_bounding_box.maxPoint.y)/2,
            (self._cached_bounding_box.minPoint.z + self._cached_bounding_box.maxPoint.z)/2)

    def place(self, x=_null_vector, y=_null_vector, z=_null_vector):
        transform = Matrix3D.create()
        transform.translation = Vector3D.create(x.x, y.y, z.z)
        self._local_transform.transformBy(transform)
        self._reset_cache()
        return self

    def __neg__(self):
        return Place(self.min())

    def __pos__(self):
        return Place(self.max())

    def __invert__(self):
        return Place(self.mid())

    def _reset_cache(self):
        self._cached_bodies = None
        self._cached_bounding_box = None
        self._cached_world_transform = None
        for component in self.children():
            component._reset_cache()

    def _get_world_transform(self) -> Matrix3D:
        if self._cached_world_transform is not None:
            return self._cached_world_transform
        transform = self._local_transform.copy()
        if self.parent is not None:
            transform.transformBy(self.parent._get_world_transform())
        self._cached_world_transform = transform
        return transform


class Box(Component):
    _poz_x = Vector3D.create(1, 0, 0)
    _poz_y = Vector3D.create(0, 1, 0)

    def __init__(self, x: float, y: float, z: float, name: str = None):
        super().__init__(name)
        self.x = x
        self.y = y
        self.z = z
        self.body = brep().createBox(OrientedBoundingBox3D.create(
            Point3D.create(x/2, y/2, z/2),
            Box._poz_x, Box._poz_y,
            x, y, z))

    def _raw_bodies(self):
        return [self.body]

    def _default_name(self):
        return "Box"


class Union(Component):
    def __init__(self, *components: Component, name=None):
        super().__init__(name)
        result_body = None
        self._children = []
        for component in components:
            if component.parent is not None:
                # TODO: need to make a copy
                pass
            for body in component.bodies():
                if result_body is None:
                    result_body = brep().copy(body)
                else:
                    brep().booleanOperation(result_body, body, adsk.fusion.BooleanTypes.UnionBooleanType)
            component._parent = self
            self._children.append(component)
        self._body = result_body

    def _raw_bodies(self) -> Iterable[adsk.fusion.BRepBody]:
        return [self._body]
        pass

    def children(self) -> Iterable['Component']:
        return tuple(self._children)

    def _default_name(self):
        return "Union"


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
