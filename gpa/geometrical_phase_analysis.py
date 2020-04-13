import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs
import pint

from gpa.utils import relative2value


# TODO
# - check two g are perpendicular, if not do something about it
# - speeding up phase calculation
# - sync radius ROI
# - add support for no perpendicular g
# - crop fft?
# - add peak finder to set automatically roi to the two frequencies (or other)
# - visualise power_spectrum with apodization but perform calculation without


class GeometricalPhaseAnalysis:

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

    def set_phase(self, phase_g1, phase_g2):
        self.phases['g1'] = phase_g1
        phase_g1.metadata.General.title = 'g1'
        self.phases['g2'] = phase_g2
        phase_g2.metadata.General.title = 'g2'

    def g_vectors(self, calibrated=True):
        return {g:self._g_vector(g, calibrated=calibrated)
                for g in ['g1', 'g2']}

    def _g_vector(self, g, calibrated=True):
        if calibrated:
            cal = np.array([1, 1])
        else:
            cal = np.array([axis.scale for axis in
                            self.signal.axes_manager.signal_axes])

        return np.array([self.rois[g].cx / cal[0],
                         self.rois[g].cy / cal[1]])

    @property
    def g_vectors_norm(self):
        return {g:self._get_g_vector_norm(g) for g in ['g1', 'g2']}

    def _get_g_vector_norm(self, g, calibrated=True):
        g_vector = self._get_g_vector(g, calibrated=calibrated)
        return np.sqrt(g_vector[0]**2 + g_vector[1]**2)

    def set_fft(self, *args, **kwargs):
        self.fft = self.signal.fft(*args, **kwargs)
        # signal_axes = fft.axes_manager.signal_axes
        # start = [(axis.axis[-1] - axis.axis[0]) / 4 + axis.axis[0] for
        #          axis in signal_axes]
        # end = [(axis.axis[-1] - axis.axis[0]) * 3 / 4 + axis.axis[0] for
        #        axis in signal_axes]
        # self.fft = fft.isig[start[0]:end[0], start[1]:end[1]]

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
        >>> gp_analysis = gpa.GeometricalPhaseImage(s)
        >>> gp_analysis.set_fft(True)
        >>> gp_analysis.plot_fft(True)

        Specify the ROI arguments (see CircleROI documentation)

        >>> gp_analysis.add_rois([[4.2793, 1.1138, 2.0],
        ...                       [-1.07473, 4.20123, 2.0]])

        Specify a spatial frequency

        >>> gp_analysis.add_rois(0.6)

        """
        # do something more clever when we have the peak finder
        # in the mean time, set roi at 2 A
        if roi_args is None:
            ureg = pint.UnitRegistry()
            value = 2.0 * ureg.angstrom
            value = value.to(self.signal.axes_manager[0].units).magnitude
            roi_args = 1 / value
        if isinstance(roi_args, float):
            roi_args = [[roi_args, 0, 2],
                        [0, roi_args, 2]]
        for key, args in zip(['g1', 'g2'], roi_args):
            self.add_roi(key, *args)

    def calculate_phase(self):
        phase_g1 = self.fft.get_phase_from_roi(self.rois['g1'], reduced=True)
        phase_g2 = self.fft.get_phase_from_roi(self.rois['g2'], reduced=True)

        self.set_phase(phase_g1, phase_g2)

    def plot_phase(self, add_refinement_rois=True):
        self.phases['g1'].plot(cmap='viridis')
        self.phases['g2'].plot(cmap='viridis')

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

    def unwrap_phase(self):
        for key, phase in self.phases.items():
            phase.unwrap()

    def calculate_displacement(self):
        """
        Calculate the displacement maps along the x and y axis from the phase
        images

        Parameters
        ----------
        phase_g1, phase_g2 : :py:class:`~gpa.atomic_resolution.GeometricPhaseImage`
            Phase images corresponding to vectors g1 and g2.

        Returns
        -------
        u_x, u_y : np.ndarray of dimension 2
            Displacement map along the x and y axis

        """
        phase_stack = np.vstack([self.phases['g1'].data.flatten(),
                                 self.phases['g2'].data.flatten()])

        U_stack = self._a_matrix() @ phase_stack / (-2*np.pi)

        u_x = U_stack[0].reshape(self.phases['g1'].data.shape)
        u_y = U_stack[1].reshape(self.phases['g2'].data.shape)

        self.u_x = hs.signals.Signal2D(u_x)
        self.u_x.metadata.Signal.quantity = "$u_{x}$"

        self.u_y = hs.signals.Signal2D(u_y)
        self.u_y.metadata.Signal.quantity = "$u_{y}$"

        return self.u_x, self.u_y

    def calculate_strain(self):
        """
        Calculate the strain tensor from the phase image.

        Notes
        -----
        See equation (42) in Hytch et al. Ultramicroscopy 1998

        """
        # Calculate the derivative the phase image
        gradient_phase_g1 = self.phases['g1'].gradient(flatten=True)
        gradient_phase_g2 = self.phases['g2'].gradient(flatten=True)

        # Make the matrix of the derivative of the phase
        gradient_phases = np.stack([gradient_phase_g1, gradient_phase_g2])

        # Multiply both matrix
        e = self._a_matrix() @ gradient_phases / (-2*np.pi)

        shape = self.phases['g1'].data.shape
        e_xx, e_xy = e[0, 0].reshape(shape), e[0, 1].reshape(shape)
        e_yx, e_yy = e[1, 0].reshape(shape), e[1, 1].reshape(shape)

        self.e_xx = hs.signals.Signal2D(e_xx)
        self.e_xx.metadata.General.title = "$\epsilon_{xx}$"
        self.e_xx.metadata.Signal.quantity = "$\epsilon_{xx}$ (%)"

        self.e_yy = hs.signals.Signal2D(e_yy)
        self.e_yy.metadata.General.title = "$\epsilon_{yy}$"
        self.e_yy.metadata.Signal.quantity = "$\epsilon_{yy}$ (%)"

        self.e_xy = hs.signals.Signal2D(e_xy)
        self.e_xy.metadata.General.title = "$\epsilon_{xy}$"
        self.e_xy.metadata.Signal.quantity = "$\epsilon_{xy}$ (%)"

        self.e_yx = hs.signals.Signal2D(e_yx)
        self.e_yx.metadata.General.title = "$\epsilon_{yx}$"
        self.e_yx.metadata.Signal.quantity = "$\epsilon_{yx}$ (%)"

        self.theta = hs.signals.Signal2D(0.5*(e_xy + e_yx))
        self.theta.metadata.General.title = "$\theta$"
        self.theta.metadata.Signal.quantity = "$\theta$ (%)"

        self.omega = hs.signals.Signal2D(0.5*(e_xy - e_yx))
        self.omega.metadata.General.title = "$\omega$"
        self.omega.metadata.Signal.quantity = "$\omega$ (%)"

    def plot_strain(self, components='all', same_figure=True, **kwargs):
        if 'cmap' not in kwargs.keys():
            kwargs['cmap'] = 'viridis'
        if components == 'all':
            components = ['e_xx', 'e_yy', 'e_xy', 'e_yx']
        elif isinstance(components, str):
            components = [components]
        if same_figure:
            signals = [getattr(self, component) for component in components]
            hs.plot.plot_images(signals, cmap='viridis', vmin=-0.01, vmax=0.01,
                                per_row=2, label='titles', colorbar='single',
                                scalebar=[2], axes_decor=None)
        else:
            if 'e_xx' in components:
                self.e_xx.plot(**kwargs)
            if 'e_yy' in components:
                self.e_yy.plot(**kwargs)
            if 'e_xy' in components:
                self.e_xy.plot(**kwargs)
            if 'e_yx' in components:
                self.e_yx.plot(**kwargs)

    def _a_matrix(self, norm=False):
        """
        The a matrix is the matrix formed of the vector a1 and a2, which
        correspond to the lattice in real space defined by the reciprocal
        lattice vectors g1 and g2

        Notes
        -----
        See equation (36) in Hytch et al. Ultramicroscopy 1998
        """
        a_matrix = np.linalg.inv(self._g_matrix().T)

        return a_matrix

    def _g_matrix(self):
        """
        The g matrix is the matrix formed of the vector g1 and g2
        """
        # get the g1, g2 vectors in pixel
        g_matrix = np.array(list(self.g_vectors(calibrated=False).values()))
        # normalise then to the size of the image
        g_matrix = g_matrix / np.array(self.signal.data.shape)

        return g_matrix