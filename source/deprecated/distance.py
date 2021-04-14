# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
"""
Distance analysis tools --- :mod:`pmda.distances`
=======================================

This module contains parallel versions of analysis tasks in
:mod:`MDAnalysis.analysis.distances`.

.. autoclass:: RMSD
   :members:
   :undoc-members:
   :inherited-members:

"""
from __future__ import absolute_import

from MDAnalysis.analysis import distances
import numpy as np

from .parallel import ParallelAnalysisBase


class Distance(ParallelAnalysisBase):
    """Parallel distances analysis.

    Calculate the pairwise distance between
    :class:`~MDAnalysis.core.groups.AtomGroup` `sel1` and `sel2`.
    The single frame calculation is performed with
    :func:`MDAnalysis.analysis.distances.distance_array`.

    Attributes
    ----------
    distance : array
         Result as a NxM array where each row contains
         `[frame, time (ps), distance]`.


    Parameters
    ----------
    sel1 : AtomGroup
         atoms of the sel1 group.
    sel2 : AtomGroup
         atoms of the sel2 group



    Note
    ----
   D.distance_array(sel1.positions, sel2.positions, box=u.dimensions) for ts in u.trajectory[start:stop:dt]], decimals=3)


    """
    def __init__(self, sel1, sel2, dt, u):
        universe = mobile.universe
        super(Distance, self).__init__(u, (sel1, ))
        self._ref_pos = ref.positions.copy()
        self.superposition = superposition

    def _prepare(self):
        self.rmsd = None

    def _conclude(self):
        self.rmsd = np.vstack(self._results)

    def _single_frame(self, self.dt, self.sel1, self.sel2):
        return (distances.distance_array(sel1.positions, sel2.positions, box=u.dimensions)
