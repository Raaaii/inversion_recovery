#!/usr/bin/python
"""
Exponential Fitting Module
Author: Rajka Pejanovic, WCOLEN Normale Supérieure
Description:
    This module implements the ExponentialFitting class for performing
    exponential fitting on experimental data using least-squares minimization.

    Two models are supported:

    1. Mono-Exponential Model:
         I(x) = M_inf - M0 * exp(-x / T1)
       where:
         - M0 is the amplitude of the decaying exponential,
         - T1 is the relaxation time,
         - M_inf is the baseline offset.

    2. Bi-Exponential Model:
         I(x) = A1 * exp(-x / T1) + A2 * exp(-x / T2) + Y0
       where:
         - A1 and T1 represent the amplitude and time constant of the first component,
         - A2 and T2 represent the amplitude and time constant of the second component,
         - Y0 is the baseline offset.

    The class leverages the lmfit library for parameter estimation and
    produces fit reports and comparison plots.

Usage:
    from inv_rec_fit import ExponentialFitting
    # Provide time delays (x) and measured intensities (y)
    fitter = ExponentialFitting(time, intensities)

    # For mono-exponential fitting:
    fitter.fit_mono()

    # For bi-exponential fitting:
    fitter.fit_bi()
"""

from math import inf
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Minimizer, Parameters, fit_report


class ExponentialFitting:
    """
    A class to perform exponential fitting on experimental data.

    This class supports both mono-exponential and bi-exponential models.

    **Mono-Exponential Model:**

        I(x) = M_inf - M0 * exp(-x / T1)

    where:
      - M0: amplitude parameter for the exponential decay.
      - T1: relaxation time.
      - M_inf: baseline (offset) parameter.

    **Bi-Exponential Model:**

        I(x) = A1 * exp(-x / T1) + A2 * exp(-x / T2) + Y0

    where:
      - A1: amplitude for the first exponential component.
      - T1: time constant for the first component.
      - A2: amplitude for the second exponential component.
      - T2: time constant for the second component.
      - Y0: baseline offset.

    Attributes:
      intensities (np.ndarray): The observed intensity data.
      time (np.ndarray): The independent variable (e.g. inversion delays).
    """

    def __init__(
        self,
        time,
        intensities,
        A1=1.0,
        A1v=True,
        A1min=0,
        A1max=inf,
        T1=1.0,
        T1v=True,
        T1min=0,
        T1max=inf,
        Y0=0.0,
        Y0v=True,
        Y0min=-inf,
        Y0max=inf,
        A2=0.0,
        A2v=False,
        A2min=0,
        A2max=inf,
        T2=0.0,
        T2v=False,
        T2min=0,
        T2max=inf,
    ):
        """
        Initialize the ExponentialFitting class.

        Parameters:
          time (array-like): The independent variable values.
          intensities (array-like): The dependent variable values (data).
          A1, T1, Y0, A2, T2: Initial guess values for the parameters.
          A1v, T1v, Y0v, A2v, T2v: Boolean flags whether the corresponding parameter should vary during the fit.
          A1min, A1max, etc.: Bounds for the parameters.
        """
        self.intensities = np.array(intensities, dtype=np.float32)
        self.time = np.array(time, dtype=np.float32)

        # Store mono exponential initial parameters.
        self.A1, self.A1v, self.A1min, self.A1max = A1, A1v, A1min, A1max
        self.T1, self.T1v, self.T1min, self.T1max = T1, T1v, T1min, T1max
        self.Y0, self.Y0v, self.Y0min, self.Y0max = Y0, Y0v, Y0min, Y0max

        # Store bi exponential initial parameters.
        self.A2, self.A2v, self.A2min, self.A2max = A2, A2v, A2min, A2max
        self.T2, self.T2v, self.T2min, self.T2max = T2, T2v, T2min, T2max

    def MonoExpResidual(self, params, x, data=None):
        """
        Compute the residual for a mono-exponential model.

        The model is:

            I(x) = M_inf - M0 * exp(-x / T1)

        Parameters:
          params (Parameters): lmfit Parameters containing M0, T1, and M_inf.
          x (array-like): Independent variable values.
          data (array-like, optional): If provided, the residual computed is model - data.

        Returns:
          np.ndarray: The residual values.
        """
        M0 = params["M0"]
        T1 = params["T1"]
        M_inf = params["M_inf"]
        model = M_inf - M0 * np.exp(-x / T1)
        if data is None:
            return model
        return model - data

    def fit_mono(self, M0=1.0, M0v=True, T1=1.0, T1v=True, M_inf=0.0, Minfv=True):
        """
        Fit the mono-exponential model to the data.

        This method sets up the parameters and uses a least-squares minimization
        to fit the model I(x) = M_inf - M0 * exp(-x / T1).

        Parameters:
          M0, T1, M_inf: Initial guess values for the parameters.
          M0v, T1v, Minfv: Boolean flags indicating if the parameter is varied during fitting.

        After fitting, the fit result is stored in self.fit_result and a text report is generated.
        Also a plot of the data and fit is created.
        """
        params = Parameters()
        params.add("M0", value=M0, vary=M0v)
        params.add("T1", value=T1, vary=T1v, min=0)
        params.add("M_inf", value=M_inf, vary=Minfv)

        minimizer = Minimizer(
            self.MonoExpResidual,
            params,
            fcn_args=(self.time,),
            fcn_kws={"data": self.intensities},
        )
        out = minimizer.minimize(method="nelder")

        self.fit_result = out  # Store fit result for later access

        if out.success:
            fit_curve = self.MonoExpResidual(out.params, self.time)
            self.report = fit_report(out)
            T1_val = out.params["T1"].value
            T1_err = out.params["T1"].stderr

            print(self.report)
            plt.figure("Mono-Exp Fitting")
            plt.plot(self.time, self.intensities, "o", color="red", label="Data")
            plt.plot(self.time, fit_curve, "-", color="black", label="Mono-Exp Fit")
            textstr = f"T₁ = {T1_val:.3f} ± {T1_err:.3f} s"
            plt.text(
                0.05,
                0.95,
                textstr,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(facecolor="white", edgecolor="black"),
            )
            plt.axhline(0, color="k", linewidth=0.5)
            plt.xlabel("Inversion Delay (s)")
            plt.ylabel("Signal Intensity")
            plt.legend()
            plt.tight_layout()
        else:
            print("Mono-exponential fit failed.")

    def BiExpResidual(self, params, x, data=None):
        """
        Compute the residual for a bi-exponential model.

        The model is:

            I(x) = A1 * exp(-x / T1) + A2 * exp(-x / T2) + Y0

        Parameters:
          params (Parameters): lmfit Parameters containing A1, T1, A2, T2, and Y0.
          x (array-like): Independent variable values.
          data (array-like, optional): If provided, returns model - data.

        Returns:
          np.ndarray: The residual values.
        """
        A1 = params["A1"]
        T1 = params["T1"]
        Y0 = params["Y0"]
        A2 = params["A2"]
        T2 = params["T2"]

        model = A1 * np.exp(-x / T1) + A2 * np.exp(-x / T2) + Y0
        if data is None:
            return model
        return model - data

    def fit_bi(self, A1=1.0, T1=1.0, Y0=0.0, A2=1.0, T2=1.0):
        """
        Fit the bi-exponential model to the data.

        This method sets up the parameters and uses a least-squares minimization
        to fit the model:

            I(x) = A1 * exp(-x / T1) + A2 * exp(-x / T2) + Y0

        Parameters:
          A1, T1, Y0, A2, T2: Initial guess values for the corresponding parameters.

        After fitting, the fit result is stored in self.fit_result and a text report is generated.
        Also a plot of the data and fitted model is produced.
        """
        params = Parameters()
        params.add("A1", value=A1, vary=self.A1v, min=self.A1min, max=self.A1max)
        params.add("T1", value=T1, vary=self.T1v, min=self.T1min, max=self.T1max)
        params.add("Y0", value=Y0, vary=self.Y0v, min=self.Y0min, max=self.Y0max)
        params.add("A2", value=A2, vary=self.A2v, min=self.A2min, max=self.A2max)
        params.add("T2", value=T2, vary=self.T2v, min=self.T2min, max=self.T2max)

        minimizer = Minimizer(
            self.BiExpResidual,
            params,
            fcn_args=(self.time,),
            fcn_kws={"data": self.intensities},
        )
        out = minimizer.leastsq()

        self.fit_result = out

        if out.success:
            fit_curve = self.BiExpResidual(out.params, self.time)
            self.report = fit_report(out)
            print(self.report)

            plt.figure("Bi-Exp Fitting")
            plt.plot(self.time, self.intensities, "o", color="red", label="Data")
            plt.plot(self.time, fit_curve, "-", color="black", label="Bi-Exp Fit")
            # Example annotation using T1 parameter (you might add more detailed annotations for other parameters)
            textstr = (
                f"T1 = {out.params['T1'].value:.3f} ± {out.params['T1'].stderr:.3f} s"
            )
            plt.text(
                0.05,
                0.95,
                textstr,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(facecolor="white", edgecolor="black"),
            )
            plt.axhline(0, color="k", linewidth=0.5)
            plt.xlabel("Inversion Delay (s)")
            plt.ylabel("Signal Intensity")
            plt.legend()
            plt.tight_layout()
        else:
            print("Bi-exponential fit failed.")
