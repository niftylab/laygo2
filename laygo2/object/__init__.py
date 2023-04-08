# -*- coding: utf-8 -*-
########################################################################################################################
#
# Copyright (c) 2020, Nifty Chips Laboratory, Hanyang University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#   disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#    following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################################################################

"""
**laygo2.object** package is a collection of core object classes that 
are used to represent physical layout structures and design hierarchies. 

The package consists of four modules:

- **laygo2.object.physical**: This module defines classes for physical objects, which composes the actual IC layout.

- **laygo2.object.template**: This module describes classes for templates, which generate various instance objects based on the target technology and design parameters.

- **laygo2.object.grid**: This module describes grid classes, which provide an abstract representation of placement and routing coordinates and parameters.

- **laygo2.object.database**: This module implements classes for design hierarchy management, enabling users to manage and maintain the relationships between design elements.

The following UML diagram of the object package provides a visual representation of the relationships between the four modules and their subclasses.

.. image:: ../assets/img/user_guide_uml.png

Check the following links for the details of the modules and their subclasses.
"""

#from laygo2.object import *
from . import *
from .database import *
from .physical import *
# template packages
from .template import *
from .template.core import *
from .template.routing import *
# grid packages
from .grid import *
from .grid.core import *
from .grid.placement import *
from .grid.routing import *
