# NMR Inversion Recovery Analysis

A Python module for processing and analyzing 2D NMR inversion recovery data acquired from Bruker/TopSpin instruments.

## Author

**Rajka Pejanovic**

## Description

This package processes 2D NMR data using the `Proc2D` and `ExponentialFitting` classes. It includes:

- Reading and processing raw Bruker NMR data.
- Interactive peak selection with `matplotlib` widgets.
- Computation and visualization of:
  - 2D contour plots
  - Delay spectra
  - T₁ relaxation fitting (mono- and bi-exponential)
- Interactive HTML plots with `Plotly`.
- Customizable units (Hz vs ppm) and fit types.

## Structure

