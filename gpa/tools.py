# BSD 3-Clause License
#
# Copyright (c) 2020, Eric Prestat
# All rights reserved.

import warnings

import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs
from hyperspy.roi import BaseROI, RectangularROI
import pint

from gpa.drawing import add_vector_basis, add_roi_to_signal_plot
from gpa.utils import relative2value, rotation_matrix, rotate_strain_tensor


# TODO:
# - refactor phase refinement/phase calculation to be able to update the phase
#   in place.
# - Mask area where the amplitude is very low (means noise)


class GeometricalPhaseAnalysisTool:

    def __init__(self, signal, synchronise_roi_radius=True):
        self.signal = signal
        self.fft_signal = None
        self.rois = {}
        self.refinement_roi = None
        self.phases = {}
        self.amplitudes = {}
        self._synchronise_roi_radius = synchronise_roi_radius

        self.angle = None

        self.u_x = None
        self.u_y = None

        self.e_xx = None
        self.e_xy = None
        self.e_yy = None
        self.e_yx = None
        self.theta = None
        self.omega = None

    @property
    def synchronise_roi_radius(self):
        return self._synchronise_roi_radius

    @synchronise_roi_radius.setter
    def synchronise_roi_radius(self, value):
        self._synchronise_roi_radius = value
        for roi in self.rois.values():
            if value:
                roi.events.changed.connect(self._sync_radius_roi, {'roi': 'roi'})
            else:
                for f in roi.events.changed.connected:
                    roi.events.changed.disconnect(f)

    @property
    def spatial_resolution(self):
        """
        Returns the estimated spatial resolution in calibrated units.

        Returns
        -------
        float
            The estimated spatial resolution in calibrated units.

        Notes
        -----
        For consideration on spatial resolution, see
        Rouviere & Sarigiannidou Ultramicroscopy 106 (2005) 1-17

        """
        # When using a Gaussian mask, we the assume that the spatial resolution
        # is 3/(2pi*sigma) where sigma is the sigma of the Gaussian mask in
        # reciprocal space
        # Here the sigma of the gaussian is sigma
        # See Rouviere & Sarigiannidou Ultramicroscopy 106 (2005) 1-17
        if len(self.rois) == 0:
            raise ValueError("The ROI(s) need to be added first.")
        if not self.synchronise_roi_radius:
            raise ValueError('The radius of the ROIs needs to be synchronised '
                             'to determine a meaningfull spatial resolution. '
                             'Use the `synchronise_roi_radius` attribute to '
                             'True to synchronise the radius of the ROIs.')
        return 3 / (2*np.pi) / self.rois['g1'].r

    @spatial_resolution.setter
    def spatial_resolution(self, value):
        """
        Set the radius of the ROI in reciprocal space to achieve a specific
        spatial resolution. See references provided in the Notes for more
        details.

        Parameters
        ----------
        value : float
            Desired spatial resolution.

        Returns
        -------
        None.

        Notes
        -----
        For consideration on spatial resolution, see
        Rouviere & Sarigiannidou Ultramicroscopy 106 (2005) 1-17

        """
        if len(self.rois) == 0:
            raise ValueError("The ROI(s) need to be added first.")
        spatial_resolution2sigma = lambda v: 3 / (2*np.pi) / v
        rois = list(self.rois.values())
        rois[0].r = spatial_resolution2sigma(value)
        _plot = self._get_fft_plot()
        # ROI are not sync or plot is not active
        if not self.synchronise_roi_radius or not (_plot and _plot.is_active):
            for roi in rois[1:]:
                roi.r = spatial_resolution2sigma(value)

    def g_vectors(self, calibrated=True):
        """
        g-vectors corresponding to the geometrical phase image(s) projected.
        The coordinate are defined with the respect to the horizonal and
        vertical axes.

        Parameters
        ----------
        calibrated : bool, optional
            If True, the g-vectors are in calibrated units (typically 1/nm),
            otherwise, they are in pixel normalised by the number of pixels of
            the image. The default is True.

        Returns
        -------
        dict
            Dictionary containing the g-vectors corresponding to each phase
            image.

        """
        return {g:self._g_vector(g, calibrated=calibrated)
                for g in self.rois.keys()}

    def _g_vector(self, g, calibrated=True):
        roi = self.rois[g]
        like = self.fft_signal.data
        if calibrated:
            # Use the roi values directly
            factor = np.ones(2,  like=like)
        else:
            # needs to convert to pixel value, which is used to do the
            # matrix calculation
            factor = self.fft_signal._get_g_convertion_factor(like=like)

        return np.array(roi[:2], like=like) / factor

    def set_fft(self, shift=True, **kwargs):
        """
        Calculate the FFT of the signal.

        Parameters
        ----------
        shift : bool, optional
            If ``True``, the origin of FFT will be shifted to the centre
            (default is ``False``).
        **kwargs
            The keywords argument are passed to the `fft` method of the
            hyperspy Signal2D.

        Returns
        -------
        None.

        """
        self.fft_signal = self.signal.fft(shift=True, **kwargs)

    def _get_fft_plot(self):
        if self.fft_signal is not None:
            return self.fft_signal._plot

    def plot_power_spectrum(self, **kwargs):
        """
        Plot the power spectrum of the signal. As a convenience, only the
        central part of the power spectrum is displayed.

        Parameters
        ----------
        *args, **kwargs
            The keywords argument are passed to the plot method of the
            hyperspy Signal2D.

        Returns
        -------
        None.

        """
        if self.fft_signal is None:
            self.set_fft(shift=True)

        self.fft_signal.plot(power_spectrum=True, **kwargs)
        signal_axes = self.fft_signal.axes_manager.signal_axes
        start = [relative2value(axis, 3/8) for axis in signal_axes]
        end = [relative2value(axis, 1 - 3/8) for axis in signal_axes]

        ax = self.fft_signal._plot.signal_plot.ax
        ax.set_xlim(start[0], end[0])
        # hyperspy image plotting start from top
        ax.set_ylim(-start[1], -end[1])

        for roi in self.rois.values():
            roi.add_widget(self.fft_signal,
                           axes=self.fft_signal.axes_manager.signal_axes)

    def _add_roi(self, g, *args):
        roi = hs.roi.CircleROI(*args)
        if self.fft_signal is None:
            raise RuntimeError("The Fourier Transform must be computed first.")
        if self.fft_signal._plot is not None:
            try:
                roi.interactive(self.fft_signal, snap=False)
            except TypeError:
                # HyperSpy version doesn't support snap argument
                roi.interactive(self.fft_signal)
            if self.synchronise_roi_radius:
                roi.events.changed.connect(self._sync_radius_roi, {'roi': 'roi'})

        self.rois[g] = roi

    def _sync_radius_roi(self, roi):
        for _roi in self.rois.values():
            if _roi is not roi:
                with _roi.events.changed.suppress():
                    _roi.r = roi.r

    def add_rois(self, roi_args=None):
        """
        Add the ROIs on the power spectrum to select the two g vectors. The
        radius of the ROIs defines the sigma of the Gaussian mask.

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
        >>> gpa_tool.plot_power_spectrum()

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
            self._add_roi(f'g{i}', *args)

    def remove_rois(self):
        """
        Remove the ROI(s) from the power spectrum (if displayed) and reset the
        phase and amplitude images and rois.

        Returns
        -------
        None.

        """
        _plot = self._get_fft_plot()
        if _plot and _plot.is_active:
            for roi in self.rois.values():
                for w in roi.widgets:
                    w.close(render_figure=False)
            _plot.signal_plot.render_figure()
        self.rois = {}
        self.phases = {}
        self.amplitude = {}

    def set_refinement_roi(self, roi=None):
        """
        Set the area where the phase is refined. If the phases are displayed,
        the ROI is added on the phase figure.

        Parameters
        ----------
        roi : hyperspy ROI or list of float or None, optional
            If list of float, a rectangular ROI is created and the list of
            float is used to initialised the ROI. If None, a rectangular ROI
            is set in the middle of the image. The default is None.

        Returns
        -------
        None.

        """
        if roi is None:
            roi = self._get_default_refinement_roi()
        elif not isinstance(roi, BaseROI):
            roi = hs.roi.RectangularROI(*roi)

        self.refinement_roi = roi

        for phase in self.phases.values():
            add_roi_to_signal_plot(phase, roi)

        for amplitude in self.amplitudes.values():
            add_roi_to_signal_plot(phase, roi)

    def _get_default_refinement_roi(self):
        signal_axes = self.signal.axes_manager.signal_axes
        start = [relative2value(axis, 1/4) for axis in signal_axes]
        end = [relative2value(axis, 3/4) for axis in signal_axes]

        return RectangularROI(*start, *end)

    def calculate_phase(self, unwrap=False):
        """
        Calculate the phase for each g-vector previously set.

        Returns
        -------
        None.

        """
        kwargs = {'reduced':True, 'unwrap':unwrap,
                  'also_return_amplitude': True}

        for g, roi in self.rois.items():
            res = self.fft_signal.get_phase_from_roi(roi, g, **kwargs)
            self.phases[g], self.amplitudes[g] = res

    def plot_phase(self, refinement_roi=True, **kwargs):
        """
        Plot the phase for each g-vector previously set.

        Parameters
        ----------
        refinement_roi : bool, optional
            If True, also add the refinement ROI. If no refinement ROI have
            been previously, a rectangular ROI is added in the middle of the
            image. The default is True.

        **kwargs
            The keywords argument are passed to the plot method of the hyperspy
            Signal2D.

        Returns
        -------
        None.

        """
        for phase in self.phases.values():
            if 'cmap' not in kwargs:
                kwargs['cmap'] = 'viridis'
            phase.plot(**kwargs)

            if refinement_roi and self.refinement_roi is not None:
                add_roi_to_signal_plot(phase, self.refinement_roi)

    def plot_amplitude(self, refinement_roi=True, **kwargs):
        """
        Plot the amplitude for each g-vector previously set.

        Parameters
        ----------
        refinement_roi : bool, optional
            If True, also add the refinement ROI. If no refinement ROI have
            been previously, a rectangular ROI is added in the middle of the
            image. The default is True.

        **kwargs
            The keywords argument are passed to the plot method of the hyperspy
            Signal2D.

        Returns
        -------
        None.

        """
        for amplitude in self.amplitudes.values():
            if 'cmap' not in kwargs:
                kwargs['cmap'] = 'viridis'
            amplitude.plot(**kwargs)

            if refinement_roi and self.refinement_roi is not None:
                add_roi_to_signal_plot(amplitude, self.refinement_roi)

    def refine_phase(self):
        """
        Refine the gradient of the phase so that the strain in the reference
        area is zero. The corresponding g-vector is adjusted accordingly.

        Returns
        -------
        None.

        Raises
        ------
        RuntimeError
            When the refinement ROI is not set. Use the `plot_phase` or the
            `set_refinement_roi` method to set them.

        """
        if self.refinement_roi is None:
            raise RuntimeError("The refinement ROI needs to be set first.")
        for phase, roi in zip(self.phases.values(), self.rois.values()):
            if phase._gradient is None:
                phase.gradient()
            g_refinement = phase.refine_phase(self.refinement_roi)
            g_refinement *= self.fft_signal._get_g_convertion_factor(
                like=self.fft_signal.data
                )
            roi.cx -= g_refinement[0]
            roi.cy -= g_refinement[1]

    def calculate_displacement(self, angle=None):
        """
        Calculate the displacement maps along the x and y axis from the phase
        images.

        Parameters
        ----------
        angle : float or None, optional
            Set the angle of the x vector relative to the horizontal axis

        Returns
        -------
        u_x, u_y : np.ndarray of dimension 2
            Displacement map along the x and y axis

        """
        if angle is not None:
            self.angle = angle

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

    def _get_grad_phase_array(self):
        phase_grad = []
        # Calculate the derivative of the phase image
        for phase in self.phases.values():
            if phase._gradient is None:
                phase.gradient()
            with phase._gradient.unfolded():
                phase_grad.append(phase._gradient.data)

        shape = self.signal.axes_manager.signal_shape
        # if only one g, append nul phase
        if len(phase_grad) == 1:
            phase_grad.append([np.zeros(np.multiply(*shape)) for i in range(2)])

        return np.array(phase_grad, like=self.phases['g1'].data)

    def calculate_strain(self, angle=None):
        """
        Calculate the strain tensor from the phase image. The strain components
        are set as attributes with the name: `e_xx`, `e_yy`, `theta` and `omega`

        Parameters
        ----------
        angle : float or None, optional
            Set the angle of the x vector relative to the horizontal axis

        Notes
        -----
        See equation (42) in Hytch et al. Ultramicroscopy 1998

        """
        from gpa.signals import StrainComponent

        e = self._a_matrix() @ self._get_grad_phase_array() / (-2*np.pi)

        if angle is not None:
            self.angle = angle
            e = rotate_strain_tensor(angle, e[0, 0], e[1, 1], e[1, 0], e[0, 1],
                                     like=self.fft_signal.data)

        shape = self.signal.axes_manager.signal_shape[::-1]
        e_xx = e[0, 0].T.reshape(shape)
        e_yy = e[1, 1].T.reshape(shape)
        e_yx = e[1, 0].T.reshape(shape)
        e_xy = e[0, 1].T.reshape(shape)

        axes_list = list(self.signal.axes_manager.as_dictionary().values())
        self.e_xx = StrainComponent(e_xx, axes=axes_list)
        self.e_xx.metadata.General.title = r"$\epsilon_{xx}$"
        self.e_xx.metadata.Signal.quantity = r"$\epsilon_{xx}$"

        self.e_yy = StrainComponent(e_yy, axes=axes_list)
        self.e_yy.metadata.General.title = r"$\epsilon_{yy}$"
        self.e_yy.metadata.Signal.quantity = r"$\epsilon_{yy}$"

        self.theta = StrainComponent(0.5*(e_xy + e_yx), axes=axes_list)
        self.theta.metadata.General.title = r"$\theta$"
        self.theta.metadata.Signal.quantity = r"$\theta$"

        self.omega = StrainComponent(0.5*(e_xy - e_yx), axes=axes_list)
        self.omega.metadata.General.title = r"$\omega$"
        self.omega.metadata.Signal.quantity = r"$\omega$"

        for name in ['e_xx', 'e_yy', 'theta', 'omega']:
            component = getattr(self, name)
            component.original_metadata.g_vectors = self._g_matrix(angle=angle)


    def plot_strain(self, components=None, same_figure=True, **kwargs):
        """
        Convenient method to plot the strain maps.

        Parameters
        ----------
        components : list of string or None, optional
            Name of the strain component ('e_xx', 'eyy', etc.) to plot. If None,
            the 'e_xx', 'e_yy' and 'omega' strain component will be plotted.
            The default is None.
        same_figure : bool, optional
            If True, plot the strain components on the same figure.
            The default is True.
        **kwargs
            If same_figure=True, the keyword argument are passed to the
            hs.plot.plot_images hyperspy method, Othewise, they are passed to
            the plot method of Signal2D.

        Returns
        -------
        None.

        """
        default_values = {'cmap':'viridis',
                          'vmin':'1th',
                          'vmax':'99th',
                          }
        if same_figure:
            default_values.update({'per_row': 3,
                                   'colorbar': 'single',
                                   'scalebar': [0],
                                   'axes_decor': None})

        # Set default values
        for key, value in default_values.items():
            if key not in kwargs.keys():
                kwargs[key] = value

        if components is None:
            components = ['e_xx', 'e_yy', 'omega']
        elif isinstance(components, str):
            components = [components]

        vector_basis = self._g_matrix(normalised=True)
        if same_figure:
            signals = [getattr(self, component) for component in components]
            fig = kwargs.get('fig', plt.figure(figsize=(12, 4.8)))
            axs = hs.plot.plot_images(signals, fig=fig, **kwargs)
            add_vector_basis(vector_basis, ax=axs[-1], labels=['x', 'y'],
                             animated=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                plt.tight_layout(rect=[0, 0, 0.9, 1])
        else:
            for component in components:
                s = getattr(self, component)
                s.plot(**kwargs)
                add_vector_basis(vector_basis, ax=s._plot.signal_plot.ax,
                                 labels=['x', 'y'])

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
        return np.linalg.inv(self._g_matrix(calibrated, angle=0).T)

    def _g_matrix(self, calibrated=False, normalised=False, angle=None):
        """ The g matrix is the matrix formed of the vector g1 and g2. If g2
        is undefined, we set g2 orthogonal to g1
        """
        like = self.fft_signal.data
        g_vectors = list(self.g_vectors(calibrated=calibrated).values())

        if len(g_vectors) == 1:
            g1 = g_vectors[0]
            g_vectors.append([g1[1], -g1[0]])

        g_matrix = np.array(g_vectors, like=like).T

        if angle is not None:
            g_matrix = g_matrix @ rotation_matrix(angle, like=like)
        elif self.angle is not None:
            g_matrix = g_matrix @ rotation_matrix(self.angle, like=like)

        if normalised:
            g_matrix = g_matrix / np.linalg.norm(g_matrix)

        return g_matrix
