#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 16:09:25 2021

@author: eric
"""

from hyperspy.roi import BaseROI


def add_roi_to_signal_plot(signal, roi, snap=True):
    """
    Add a roi to the figure to define the refinement area.

    Parameters
    ----------
    roi : ROI
        ROI defining the refinement area.

    """
    if not isinstance(roi, BaseROI):
        raise ValueError("A valid hyperspy ROI must be provided. "
                         f"Provided ROI: {roi}")

    if signal._plot is not None and signal._plot.is_active:
        try:
            roi.add_widget(signal, signal.axes_manager.signal_axes, snap=snap)
        except TypeError:
            # HyperSpy version doesn't support snap argument
            roi.add_widget(signal, signal.axes_manager.signal_axes)
