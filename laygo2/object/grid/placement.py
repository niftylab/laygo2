#!/usr/bin/python
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

from .core import Grid

class PlacementGrid(Grid):
    """
    PlacementGrid class implements a grid for placement of Instance and
    VirtualInstance objects.

    Notes
    -----
    **(Korean)** PlacementGrid 클래스는 Instance 및 VirtualInstance 개체들의
        배치를 위한 격자 그리드를 구현한다.

    """

    type = "placement"
    """ Type of grid. Should be 'placement' for placement grids."""

    def place(self, inst, mn):
        """
        Place the instance on the specified coordinate mn, on this grid.

        Parameters
        ----------
        inst : laygo2.object.physical.Instance or laygo2.object.physical.VirtualInstance
            The instance to be placed on the grid.
        mn : numpy.ndarray or list
            The abstract coordinate [m, n] to place the instance.

        Returns
        -------
        laygo2.object.physical.Instance or
        laygo2.object.physical.VirtualInstance :
            The placed instance.

        Example
        -------
        >>> import laygo2
        >>> from laygo2.object.grid import OneDimGrid, PlacementGrid
        >>> from laygo2.object.physical import Instance
        >>> #
        >>> # Create a grid (not needed if laygo2_tech is set up).
        >>> #
        >>> gx  = OneDimGrid(name="gx", scope=[0, 20], elements=[0])
        >>> gy  = OneDimGrid(name="gy", scope=[0, 100], elements=[0])
        >>> g   = PlacementGrid(name="test", vgrid=gx, hgrid=gy)
        >>> #
        >>> # Create an instance
        >>> #
        >>> i0 = Instance(libname="tlib", cellname="t0", name="I0", xy=[0, 0])
        >>> print(inst0.xy)
        [100, 100]
        >>> #
        >>> # Place the created instance
        >>> #
        >>> g.place(inst=i0, mn=[10,10])
        >>> # Print parameters of the placed instance.
        >>> print(i0.xy)
        [200, 1000]

        Notes
        -----
        **(Korean)** 인스턴스 xy속성에 추상좌표를 매핑함.
            파라미터
            - inst(laygo2.physical.instance): 배치할 인스턴스
            - mn(numpy.ndarray or list): 인스턴스를 배치할 추상좌표
            반환값
            - laygo2.physical.instance: 좌표가 수정된 인스턴스
        """
        inst.xy = self[mn]
        return inst
