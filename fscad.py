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
from adsk.core import Point3D, Vector3D, Matrix3D, ObjectCollection, OrientedBoundingBox3D
from typing import Iterable

import adsk.core
import adsk.fusion
import traceback
import types
import sys


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


class Component(object):
    parent = ...  # type: Component

    def __init__(self, name: str = None):
        self.parent = None
        self.active = True
        self.transform = Matrix3D.create()
        self.name = name

    def bodies(self) -> Iterable[adsk.fusion.BRepBody]:
        pass

    def create_occurrence(self) -> adsk.fusion.Occurrence:
        return _create_component(root(), *self.bodies(), name=self.name or self._default_name())

    def _get_world_transform(self) -> Matrix3D:
        transform = self.transform.copy()
        if self.parent is not None:
            transform.transformBy(self.parent._get_world_transform())
        return transform

    def _default_name(self) -> str:
        return "Component"


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
        self._cached_transform = None
        self._cached_body = None

    def bodies(self):
        world_transform = self._get_world_transform()
        if world_transform == self._cached_transform:
            return self._cached_body

        body_copy = brep().copy(self.body)
        brep().transform(body_copy, world_transform)
        self._cached_transform = world_transform
        self._cached_body = body_copy
        return [body_copy]

    def _default_name(self):
        return "Box"


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
