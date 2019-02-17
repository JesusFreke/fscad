##### fscad - A framework for programmatic CAD in Fusion 360

fscad is a wrapper around Fusion 360's apis that provide a framework for doing programmatic CAD. 

fscad is heavily inspired by OpenSCAD and its api provides a similar design paradigm, although the
syntax is obviously different and it's backed by an actual procedural language (Python). The basic
design process is that of programmatically creating basic 2d and 3d shapes and then combining them
with boolean operations.

There were a number of annoyances that I had with OpenSCAD, and I've attempted to improve on them
with fscad.

* **Speed** - fscad scripts are generally much faster to run, which allow for a faster turnaround in
the "tweak design -> view results -> tweak design -> view results..." cycle.
* **Referencing other objects** - Objects' attributes such as size, location, bounding box, etc. are
immediately available, and can be used when creating or placing a new object. For example, it's easy
to place a new object so that its "left side" is aligned with the "right side" of another object. Or
to create an object whose height matches that of some other object. I find this provides a much more
natural way to define objects, with significantly fewer "magic numbers" and less explicit math
needed.
* **Better IDE/editor** - Since fscad designs are written in Python, you can use any Python IDE to
edit them, and get the full functionality of a full-blown IDE. In particular, you can use the
fusionIdea plugin for Intellij IDEA or PyCharm to provide an easy way to launch your script in
Fusion 360 from IDEA, and even debug the script with all the power of IDEA's debugger.
* **More powerful and performant UI** - The Fusion 360 UI is much more powerful "viewer" for the
result of the design. It is also typically much faster, and doesn't get bogged down like OpenSCAD
is prone to do.
* **Language support** - You get the full power of the Python language and ecosystem of libraries
when creating your design.

#### Status
fscad is currently in "beta" status. Its API should be mostly stable, but I do reserve the option
of making breaking changes to the API. However, any such changes should be fairly minor and easy to
fix up in your design.

fscad does not expose the full functionality of Fusion 360's API. Support for various features are
typically added when I actually need them for some design I'm working on :>. If you find that there
is some functionality in Fusion 360's api you need that is not supported, feel free to file an
issue, or better, send a pull request :)

#### [Installation](https://github.com/JesusFreke/fscad/wiki/Installation)

#### [Getting Started with IntelliJ IDEA on Windows](https://github.com/JesusFreke/fscad/wiki/Getting-started-with-Intellij-IDEA-(Windows))

#### [Cheat Sheet](https://github.com/JesusFreke/fscad/wiki/CheatSheet)

#### [API Documentation](https://jesusfreke.github.io/fscad/fscad.html)

#### Example

```python
from fscad import *

def my_design():
    # Create a sphere of radius 10 cm centered at the origin
    sphere = Sphere(10)

    # Create a box that is half the size of the sphere in the x direction, and the same size of the
    # sphere in the y and z directions. The minimum point of the box starts out at the origin
    box = Box(sphere.size().x/2, sphere.size().y, sphere.size().z)

    box.place(
        # place the right edge of the box (+box) at the right edge of the sphere (+sphere)
        +box == +sphere,
        # and place the center of the box (~box) at the center of the sphere (~sphere) in the y
        # and z directions
        ~box == ~sphere,
        ~box == ~sphere)

    # Subtract the box from the sphere, leaving the left side of the sphere as a hemisphere
    hemisphere = Difference(sphere, box)
    # Nothing is actually added to the Fusion 360 document until create_occurrence() is called.
    # This is typically the last thing done on the top-level component that has been built up by
    # multiple previous operations.
    # create_children=True adds the sphere and box as hidden sub-components in the Fusion 360
    # component hierarchy. This allows you to browse through the component hierarchy to see how the
    # component was built up.
    hemisphere.create_occurrence(create_children=True)

# This is the entry point for a Fusion 360 script
def run(context):
    # This is an fscad helper function to set up a new document that will be populated by this script. It will call
    # the specified function (my_design) after creating and setting up a fresh document.
    run_design(my_design, message_box_on_error=False, document_name=__name__)
```

#### Real-World Example

To see an example of an actual, non-trivial project developed with fscad, see my
[lalboard](https://github.com/JesusFreke/lalboard/blob/master/lalboard.py) project.
fscad was developed in parallel with lalboard. I would switch back and forth between the 2
projects, adding new features to fscad as I needed them for lalboard.


Note: This is not an officially supported Google product.