# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.

from hyperspy.signals import Signal2D
import numpy as np

from gpa.drawing.vectors_basis import add_vector_basis


class StrainComponent(Signal2D):

    signal_type = 'strain_component'
    _signal_dimension = 2

    def plot(self, plot_vector_basis=True, cmap='viridis', **kwargs):
        super().plot(cmap=cmap, **kwargs)
        if plot_vector_basis:
            vector_basis = self.original_metadata.g_vectors
            vector_basis /= np.linalg.norm(vector_basis)
            add_vector_basis(vector_basis, ax=self._plot.signal_plot.ax,
                             labels=['x', 'y'])