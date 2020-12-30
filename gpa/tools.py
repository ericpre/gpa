# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.

import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs
import pint

from gpa.utils import relative2value


# TODO:
# - Refine phase
# - speeding up phase calculation?
# - sync radius ROI
# - add ROI to plot if rois already exists, when plotting fft
# - add vector basis on plot
# - rotate basis


class GeometricalPhaseAnalysisTool:

    def __init__(self, signal):
        self.signal = signal
        self.fft = None
        self.rois = {}
        self.refinement_roi = None
        self.phases = {}

        self.u_x = None
        self.u_y = None

        self.e_xx = None
        self.e_xy = None
        self.e_yy = None
        self.e_yx = None
        self.theta = None
        self.omega = None

    def set_phase(self, *phases):
        for i, phase in enumerate(phases, start=1):
            key = f'g{i} reduced Phase'
            phase.metadata.General.title = key
            self.phases[key] = phase

    def g_vectors(self, calibrated=True):
        return {g:self._g_vector(g, calibrated=calibrated)
                for g in self.rois.keys()}

    def _g_vector(self, g, calibrated=True):
        roi = self.rois[g]
        if calibrated:
            factor = 1
        else:
            axis = self.fft.axes_manager[-1]
            factor = axis.scale * axis.size

        return np.array([roi.cx, roi.cy]) / factor

    @property
    def g_vectors_norm(self):
        return {g:np.linalg.norm(g) for g in ['g1', 'g2']}

    def set_fft(self, *args, **kwargs):
        self.fft = self.signal.fft(*args, **kwargs)

    def plot_fft(self, *args, **kwargs):
        self.fft.plot(*args, **kwargs)
        ax = plt.gca()
        signal_axes = self.fft.axes_manager.signal_axes
        start = [relative2value(axis, 3/8) for axis in signal_axes]
        end = [relative2value(axis, 1 - 3/8) for axis in signal_axes]

        ax.set_xlim(start[0], end[0])
        # hyperspy image plotting start from top
        ax.set_ylim(-start[1], -end[1])

    def add_roi(self, g, *args):
        self.rois[g] = hs.roi.CircleROI(*args)
        if self.fft is None:
            raise RuntimeError("The Fourier Transform must be computed first.")
        if self.fft._plot is not None:
            self.rois[g].interactive(self.fft)

    def add_rois(self, roi_args=None):
        """
        Add the ROIs on the power spectrum to select the two g vectors.

        Parameters
        ----------
        roi_args : {str, list of list of float, None}, optional
            Parameters used to create the ROIs. If str, use this value to set
            the ROIs at this spacial frequency. If list of list of float,
            this needs to be a list of two lists of float which will be pass
            to :py:class:`hyperspy.roi.CircleROI` - see examples. If None,
            set the ROIs at a spacing frequencies corresponding to 2 Å.
            Default is None.

        Returns
        -------
        None.

        Examples
        --------
        >>> s = gpa.datasets.get_atomic_resolution_tem_signal2d()
        >>> gpa_tool = gpa.GeometricalPhaseAnalysisTool(s)
        >>> gpa_tool.set_fft(True)
        >>> gpa_tool.plot_fft(True)

        Specify the ROI arguments (see CircleROI documentation)

        >>> gpa_tool.add_rois([[4.2793, 1.1138, 2.0], [-1.07473, 4.20123, 2.0]])

        Specify a spatial frequency

        >>> gpa_tool.add_rois(0.6)

        """
        if roi_args is None:
            ureg = pint.UnitRegistry()
            value = 2.0 * ureg.angstrom
            value = value.to(self.signal.axes_manager[0].units).magnitude
            roi_args = 1 / value
        if isinstance(roi_args, float):
            roi_args = [[roi_args, 0, 2],
                        [0, roi_args, 2]]
        for i, args in enumerate(roi_args, start=1):
            self.add_roi(f'g{i}', *args)

    def calculate_phase(self):
        self.set_phase(*[self.fft.get_phase_from_roi(roi, reduced=True)
                         for roi in self.rois.values()])

    def _correct_phase(self, g):
        w, h = self.signal.data.shape
        cx, cy = self.g_vectors(calibrated=True)[g]

        # gx1 = (cx-w//2)/w
        # gy1 = (cy-h//2)/h
        # x = np.arange(w)
        # y = np.arange(h)
        # X, Y = np.meshgrid(x, y)
        # # calculate term to subtract: -2*pi*(g.r)
        # return 2*np.pi*(gx1*X+gy1*Y)

        print(cx, cy)
        gx = (cx - w // 2) / w
        gy = (cy - h // 2) / h
        print(gx, gy)

        signal_axes = self.signal.axes_manager.signal_axes
        r_x = np.linspace(-0.5, 0.5, num=w) / signal_axes[0].scale
        r_y = np.linspace(-0.5, 0.5, num=h) / signal_axes[1].scale
        R_x, R_y = np.meshgrid(r_x, r_y)
        # G_r = 2 * np.pi * ((R_x * ) + (R_y * g_vector[0]))

        return 2 * np.pi * (cx * R_x + cy * R_y)

    def plot_phase(self, add_refinement_rois=True):
        for phase in self.phases.values():
            phase.plot(cmap='viridis')

        if add_refinement_rois:
            for key, phase in self.phases.items():
                returned = phase.add_refinement_roi(self.refinement_roi)
                if self.refinement_roi is None:
                    self.refinement_roi = returned

    def refine_phase(self):
        for key, phase in self.phases.items():
            phase.refine_phase(self.fft,
                               self.rois[key],
                               self.refinement_roi
                               )
        for roi in self.rois.values():
            roi.update()

    def calculate_displacement(self):
        """
        Calculate the displacement maps along the x and y axis from the phase
        images

        Returns
        -------
        u_x, u_y : np.ndarray of dimension 2
            Displacement map along the x and y axis

        """
        shape = self.signal.axes_manager.signal_shape
        phases = [phase.data.flatten() for phase in self.phases.values()]
        # only one g, append nul phase
        if len(phases) == 1:
            phases.append(np.zeros(np.multiply(*shape)))

        phase_matrix = np.vstack(phases)
        U = self._a_matrix() @ phase_matrix / (-2*np.pi)

        self.u_x = hs.signals.Signal2D(U[0].reshape(shape))
        self.u_x.metadata.Signal.quantity = "$u_{x}$"

        self.u_y = hs.signals.Signal2D(U[1].reshape(shape))
        self.u_y.metadata.Signal.quantity = "$u_{y}$"

        return self.u_x, self.u_y

    def calculate_strain(self):
        """
        Calculate the strain tensor from the phase image.

        Notes
        -----
        See equation (42) in Hytch et al. Ultramicroscopy 1998

        """
        # Calculate the derivative of the phase image
        phase_derivative = [phase.gradient(flatten=True)
                            for phase in self.phases.values()]

        shape = self.signal.axes_manager.signal_shape
        # if only one g, append nul phase
        if len(phase_derivative) == 1:
            phase_derivative.append([np.zeros(np.multiply(*shape)) for i in range(2)])

        derivative_matrix = np.array(phase_derivative)

        E = self._a_matrix() @ derivative_matrix / (-2*np.pi)
        e_xx = E[0, 0].reshape(shape)
        e_yy = E[1, 1].reshape(shape)
        e_yx = E[1, 0].reshape(shape)
        e_xy = E[0, 1].reshape(shape)

        axes_list = list(self.signal.axes_manager.as_dictionary().values())
        self.e_xx = hs.signals.Signal2D(e_xx, axes=axes_list)
        self.e_xx.metadata.General.title = r"$\epsilon_{xx}$"
        self.e_xx.metadata.Signal.quantity = r"$\epsilon_{xx}$"

        self.e_yy = hs.signals.Signal2D(e_yy, axes=axes_list)
        self.e_yy.metadata.General.title = r"$\epsilon_{yy}$"
        self.e_yy.metadata.Signal.quantity = r"$\epsilon_{yy}$"

        self.e_xy = hs.signals.Signal2D(e_xy, axes=axes_list)
        self.e_xy.metadata.General.title = r"$\epsilon_{xy}$"
        self.e_xy.metadata.Signal.quantity = r"$\epsilon_{xy}$"

        self.e_yx = hs.signals.Signal2D(e_yx, axes=axes_list)
        self.e_yx.metadata.General.title = r"$\epsilon_{yx}$"
        self.e_yx.metadata.Signal.quantity = r"$\epsilon_{yx}$"

        self.theta = hs.signals.Signal2D(0.5*(e_xy + e_yx), axes=axes_list)
        self.theta.metadata.General.title = r"$\theta$"
        self.theta.metadata.Signal.quantity = r"$\theta$"

        self.omega = hs.signals.Signal2D(0.5*(e_xy - e_yx), axes=axes_list)
        self.omega.metadata.General.title = r"$\omega$"
        self.omega.metadata.Signal.quantity = r"$\omega$"

    def plot_strain(self, components=None, same_figure=True, **kwargs):
        # Set default value
        for key, value in zip(['cmap', 'vmin', 'vmax'],
                              ['viridis', '1th', '99th']):
            if key not in kwargs.keys():
                kwargs[key] = value
        if components is None:
            components = ['e_xx', 'e_yy', 'omega']
        elif isinstance(components, str):
            components = [components]
        if same_figure:
            signals = [getattr(self, component) for component in components]
            fig = plt.figure(figsize=(12, 4.8))
            hs.plot.plot_images(signals, per_row=3, label='titles',
                                colorbar='single', scalebar=[0],
                                axes_decor=None, fig=fig, **kwargs)
            plt.tight_layout(rect=[0, 0, 0.9, 1])
        else:
            for component in components:
                getattr(self, component).plot(**kwargs)

    def _a_matrix(self, calibrated=False):
        """
        The a matrix is the matrix formed of the vector a1 and a2, which
        correspond to the lattice in real space defined by the reciprocal
        lattice vectors g1 and g2.
        Units is in pixel

        Notes
        -----
        See equation (36) in Hytch et al. Ultramicroscopy 1998
        """
        return np.linalg.inv(self._g_matrix(calibrated).T)

    def _g_matrix(self, calibrated=False):
        """ The g matrix is the matrix formed of the vector g1 and g2. If g2
        is undefined, we set g2 orthogonal to g1
        """
        g_vectors = list(self.g_vectors(calibrated=calibrated).values())
        if len(g_vectors) == 1:
            g1 = g_vectors[0]
            g_vectors.append([g1[1], -g1[0]])

        return np.array(g_vectors).T
