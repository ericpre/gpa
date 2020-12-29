import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs
import pint

from gpa.utils import relative2value


# TODO
# - check two g are perpendicular, if not do something about it
# - speeding up phase calculation
# - add median filter to the strain maps?
# - look at the
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

    def set_phase(self, *phases):
        for i, phase in enumerate(phases, start=1):
            key = f'g{i}'
            phase.metadata.General.title = 'g1'
            self.phases[key] = phase

    def g_vectors(self, calibrated=True):
        return {g:self._g_vector(g, calibrated=calibrated)
                for g in self.rois.keys()}

    def _g_vector(self, g, calibrated=True):
        if calibrated:
            cal = np.array([1, 1])
        else:
            cal = np.array([axis.scale for axis in
                            self.fft.axes_manager.signal_axes])

        return np.array([self.rois[g].cx / cal[0],
                         self.rois[g].cy / cal[1]])

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

    def unwrap_phase(self):
        for key, phase in self.phases.items():
            phase.unwrap()

    def calculate_displacement1D(self):
        """
        Calculate the displacement maps along one axis

        Returns
        -------
        u_x : np.ndarray of dimension 2
            Displacement map along the specified axis

        """
        phase_stack = np.vstack([self.phases['g1'].data.flatten(),
                                 self.phases['g2'].data.flatten()])

        U_stack = self._a_matrix() @ phase_stack / (-2*np.pi)

        u_x = U_stack[0].reshape(self.phases['g1'].data.shape)

        self.u_x = hs.signals.Signal2D(u_x)
        self.u_x.metadata.Signal.quantity = "$u_{x}$"


        return self.u_x, self.u_y

    def calculate_displacement(self):
        """
        Calculate the displacement maps along the x and y axis from the phase
        images

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

    def calculate_strain(self, median_filter_size=5):
        """
        Calculate the strain tensor from the phase image.

        Parameters
        ----------
        median_filter_size : float, default is 5
            Size of the median filter applied to the gradient of the phase map.

        Notes
        -----
        See equation (42) in Hytch et al. Ultramicroscopy 1998

        """
        # Calculate the derivative the phase image
        dP1dx, dP1dy = self.phases['g1'].gradient(
            median_filter_size=median_filter_size
            )
        dP2dx, dP2dy = self.phases['g2'].gradient(
            median_filter_size=median_filter_size
            )

        [a1x, a2x], [a1y, a2y] = self._a_matrix()

        e_xx = -1 / (2 * np.pi) * (a1x * dP1dx + a2x * dP2dx)
        e_xy = -1 / (2 * np.pi) * (a1x * dP1dy + a2x * dP2dy)
        e_yx = -1 / (2 * np.pi) * (a1y * dP1dx + a2y * dP2dx)
        e_yy = -1 / (2 * np.pi) * (a1y * dP1dy + a2y * dP2dy)

        axes_list = list(self.signal.axes_manager.as_dictionary().values())
        self.e_xx = hs.signals.Signal2D(e_xx, axes=axes_list)
        self.e_xx.metadata.General.title = "$\epsilon_{xx}$"
        self.e_xx.metadata.Signal.quantity = "$\epsilon_{xx}$"

        self.e_yy = hs.signals.Signal2D(e_yy, axes=axes_list)
        self.e_yy.metadata.General.title = "$\epsilon_{yy}$"
        self.e_yy.metadata.Signal.quantity = "$\epsilon_{yy}$"

        self.e_xy = hs.signals.Signal2D(e_xy, axes=axes_list)
        self.e_xy.metadata.General.title = "$\epsilon_{xy}$"
        self.e_xy.metadata.Signal.quantity = "$\epsilon_{xy}$"

        self.e_yx = hs.signals.Signal2D(e_yx, axes=axes_list)
        self.e_yx.metadata.General.title = "$\epsilon_{yx}$"
        self.e_yx.metadata.Signal.quantity = "$\epsilon_{yx}$"

        self.theta = hs.signals.Signal2D(0.5*(e_xy + e_yx), axes=axes_list)
        self.theta.metadata.General.title = "$\theta$"
        self.theta.metadata.Signal.quantity = "$\theta$"

        self.omega = hs.signals.Signal2D(0.5*(e_xy - e_yx), axes=axes_list)
        self.omega.metadata.General.title = "$\omega$"
        self.omega.metadata.Signal.quantity = "$\omega$"

    def plot_strain(self, components='all', same_figure=True, **kwargs):
        # Set default value
        for key, value in zip(['cmap', 'vmin', 'vmax'],
                              ['viridis', '1th', '99th']):
            if key not in kwargs.keys():
                kwargs[key] = value
        if components == 'all':
            components = ['e_xx', 'e_yy', 'e_xy']
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
            if 'e_xx' in components:
                self.e_xx.plot(**kwargs)
            if 'e_yy' in components:
                self.e_yy.plot(**kwargs)
            if 'e_xy' in components:
                self.e_xy.plot(**kwargs)
            if 'e_yx' in components:
                self.e_yx.plot(**kwargs)

    def _a_matrix(self):
        """
        The a matrix is the matrix formed of the vector a1 and a2, which
        correspond to the lattice in real space defined by the reciprocal
        lattice vectors g1 and g2.
        Units is in pixel

        Notes
        -----
        See equation (36) in Hytch et al. Ultramicroscopy 1998
        """
        g_matrix = self._g_matrix() / np.array(self.signal.data.shape)

        return np.linalg.inv(g_matrix.T)

    def _g_matrix(self, calibrated=False):
        """
        The g matrix is the matrix formed of the vector g1 and g2
        """
        # get the g1, g2 vectors in pixel
        g_matrix = np.array(list(self.g_vectors(calibrated=calibrated).values())).T

        return g_matrix