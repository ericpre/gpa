# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.

import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox
from matplotlib.patches import FancyArrow
from matplotlib.text import Text
import matplotlib.pyplot as plt

from hyperspy.misc.utils import to_numpy


class VectorBasis:

    def __init__(self, ax, vectors, labels=None, loc=1,
                 color='black', background_color='white', alpha=0.25,
                 animated=None):
        """Add a vector basis to an image.

        Parameters
        ----------
        ax : matplotlib axes
            The axes where to draw the scale bar.
        position : tuple of float
            (0, 0) position of the basis in matplotlib coordinate
        vectors : tuple of array
            Vectors of the basis
        labels : None or tuple of string or None
            The labels for each vector
        loc : str or int
            Matplotlib loc parameter, see for example the documentation of the
            `plt.legend` function.
        color : a valid matplotlib color
            The color used for the vectors and the labels.
        background_color : a valid matplotlib color
            The color for the background.
        alpha : float
            The transparency value. Must be between 0 and 1
        animated : bool
            Set animated state.

        """
        self.ax = ax
        self.vectors = vectors
        if labels is not None and len(labels) != len(vectors):
            raise ValueError("`labels` must have the same length as `vectors.`")
        self.labels = labels
        if animated is None:
            animated = ax.get_figure().canvas.supports_blit
        self.alpha = alpha
        self.animated = animated
        self.color = color
        self.background_color = background_color

        self.plot(loc)


    def plot(self, loc):
        basis_vector = AnchoredBasisVector(self.ax.transData, self.vectors,
                                           self.labels, loc=loc,
                                           color=self.color,
                                           frameon=False)

        basis_vector.patch.set_color(self.background_color)
        basis_vector.patch.set_alpha(self.alpha)
        basis_vector.set_animated(self.animated)

        self.ax.add_artist(basis_vector)
        self.ax.figure.canvas.draw_idle()


class AnchoredBasisVector(AnchoredOffsetbox):
    def __init__(self, transform, vectors, labels, loc=1, color="black",
                 **kwargs):
        """
        Draw a horizontal bar with the size in data coordinate of the give axes.
        A label will be drawn underneath (center-aligned).

        kwargs are passed to AnchoredOffsetbox.
        """
        self.basis_vector = AuxTransformBox(transform)

        vnorm = np.average([np.linalg.norm(v) for v in vectors])
        head_width = 0.3 * vnorm
        head_length = 0.3 * vnorm

        position = np.array([0, 0])
        if labels is None:
            labels = [None] * len(vectors)

        for vector, label in zip(vectors, labels):
            v2d = np.array(vector[:2])
            arrow = FancyArrow(*position, *v2d, color=color,
                               head_width=head_width, head_length=head_length)
            self.basis_vector.add_artist(arrow)

            if label is not None:
                pos = position + v2d * 1.9
                label = Text(*pos, label,
                             horizontalalignment='center',
                             verticalalignment='center')
                self.basis_vector.add_artist(label)

        AnchoredOffsetbox.__init__(self, loc, child=self.basis_vector)


def add_vector_basis(vector_basis, ax=None, loc='upper right', labels=None,
                     scaling_factor=0.15, **kwargs):
    """
    Add a vector basis defined to a matplotlib axis.

    Parameters
    ----------
    vector_basis : numpy.ndarray
        The vector basis.
    ax : matplotlib subplot, optional
        The matplotlib subplot the basis vectors will be plotted. If None,
        the last subplot is used.
        The default is None.
    loc : str or int
        Matplotlib loc parameter, see for example the documentation of the
        `plt.legend` function.
    labels : list of string or None
        Labels of the g-vectors. The list must be of the same length as the
        number of vectors. If None, set 'g1', 'g2', etc as labels.
    scaling_factor : float
        Factor defined the width of the basis vectors relative to the
        width of the image.

    Returns
    -------
    None.

    """
    if ax is None:
        ax = plt.gca()

    width = ax.images[0].get_extent()[1] - ax.images[0].get_extent()[0]
    vectors = vector_basis * width * scaling_factor / 2

    if labels is None:
        labels = [rf'g$_{i}$' for i in range(1, len(vectors)+1)]

    VectorBasis(ax, to_numpy(vectors), labels=labels, loc=loc, **kwargs)