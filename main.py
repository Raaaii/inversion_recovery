#!/usr/bin/python
"""
NMR Data Processing Pipeline
Author: Rajka Pejanovic
Description:
    This script processes 2D NMR data acquired on Bruker/TopSpin instruments using
    the Proc2D processing module. The pipeline includes:
      - Reading and processing raw data.
      - Interactive peak selection.
      - Computation of static and interactive plots (2D contour, delay spectra, and
        T₁ fit plots) using Matplotlib and Plotly.
      - User‐selectable parameter options for F2 axis units (Hz vs. ppm) and exponential
        fitting type (mono- or bi-exponential).

    The exponential fitting routines are implemented in the ExponentialFitting class
    (see inv_rec_fit.py). This script prompts the user for choices and then saves static
    images as well as interactive HTML files for later inspection.

    Author: Rajka Pejanovic

"""
from ir_analysis.fitting import ExponentialFitting
from ir_analysis.procD import Proc2D

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import plotly.graph_objects as go

# --- Adjustable parameters and plotting functions ---


def SimpleBaselineCorrection(spectr):
    """
    Apply a simple baseline correction to a spectrum.

    The correction is computed as the average of the first and last points.

    Parameters:
      spectr (array-like): The input spectrum.

    Returns:
      np.ndarray: The baseline-corrected spectrum.
    """
    correction = (spectr[0] + spectr[-1]) / 2
    return spectr - correction


def interactive_peak_selection(axis, avg_spec):
    """
    Launch an interactive Matplotlib window for selecting peak regions.

    Drag across the displayed spectrum to select a region. When the window is closed,
    a list of selected (xmin, xmax) regions is returned.

    Parameters:
      axis (array-like): The F2 axis (in Hz or ppm).
      avg_spec (array-like): The average spectrum.

    Returns:
      list of tuple: The list of selected (xmin, xmax) regions.
    """
    selected_regions = []
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(axis, avg_spec, label="Average Spectrum")
    ax.set_xlabel("F2 (Hz or ppm)")
    ax.set_ylabel("Intensity")
    ax.set_title(
        "Interactive Peak Selection\nDrag to select a region then close the window."
    )
    ax.invert_xaxis()

    def onselect(xmin, xmax):
        region = (min(xmin, xmax), max(xmin, xmax))
        selected_regions.append(region)
        ax.axvspan(region[0], region[1], color="orange", alpha=0.3)
        center = (region[0] + region[1]) / 2
        ax.text(
            center,
            max(avg_spec) * 0.8,
            f"Region {len(selected_regions)}",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=90,
        )
        fig.canvas.draw()

    span = SpanSelector(
        ax, onselect, "horizontal", useblit=True, minspan=0.1, interactive=True
    )
    plt.legend()
    plt.tight_layout()
    plt.show()
    return selected_regions


def interactive_delay_viewer(axis, ser, vdlist):
    """
    Create an interactive Plotly viewer for delay (VD) spectra.

    The user can use a slider to view the spectrum of each delay (VD).

    Parameters:
      axis (array-like): The F2 axis (in Hz or ppm).
      ser (2D np.ndarray): The processed spectrum [# delays, # spectral points].
      vdlist (array-like): The delay values for each row in ser.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=axis,
            y=ser[0, :],
            mode="lines",
            name=f"Spectrum for VD 1, delay = {vdlist[0]:.3f} s",
        )
    )
    steps = []
    for i in range(ser.shape[0]):
        step = dict(
            method="update",
            args=[
                {
                    "x": [axis],
                    "y": [ser[i, :]],
                    "name": f"Spectrum for VD {i+1}, delay = {vdlist[i]:.3f} s",
                },
                {"title": f"Spectrum for VD {i+1}, delay = {vdlist[i]:.3f} s"},
            ],
            label=f"{i+1}",
        )
        steps.append(step)
    sliders = [
        dict(
            active=0, currentvalue={"prefix": "VD Index: "}, pad={"t": 50}, steps=steps
        )
    ]
    fig.update_layout(
        title=f"Spectrum for VD 1, delay = {vdlist[0]:.3f} s",
        xaxis_title="F2 (Hz)",
        yaxis_title="Intensity",
        sliders=sliders,
    )
    fig.update_xaxes(autorange="reversed")
    html_filename = "data/interactive_delay_spectra.html"
    fig.write_html(html_filename)
    fig.show()
    print(f"Interactive delay viewer saved as '{html_filename}'.")


def interactive_fit_viewer(all_fits, all_t1_labels):
    """
    Create an interactive Plotly viewer for T₁ fit plots.

    Allows the user to scroll through the fit results of different peaks.

    Parameters:
      all_fits (list of tuple): Each tuple is (vdlist, integrals, fitted_curve).
      all_t1_labels (list of str): Labels with T₁ values and uncertainties.
    """
    fig = go.Figure()
    # Initial plot using the first peak.
    x0, y0, yfit0 = all_fits[0]
    fig.add_trace(
        go.Scatter(x=x0, y=y0, mode="markers", name=f"Data: {all_t1_labels[0]}")
    )
    fig.add_trace(go.Scatter(x=x0, y=yfit0, mode="lines", name="Fit"))
    steps = []
    for i in range(len(all_fits)):
        x, y, yfit = all_fits[i]
        step = dict(
            method="update",
            args=[
                {
                    "x": [x, x],
                    "y": [y, yfit],
                    "name": [f"Data: {all_t1_labels[i]}", "Fit"],
                },
                {"title": f"T₁ Fit - Peak {i+1}: {all_t1_labels[i]}"},
            ],
            label=f"{i+1}",
        )
        steps.append(step)
    sliders = [
        dict(active=0, currentvalue={"prefix": "Peak: "}, pad={"t": 50}, steps=steps)
    ]
    fig.update_layout(
        title=f"T₁ Fit - Peak 1: {all_t1_labels[0]}",
        xaxis_title="Inversion Delay (s)",
        yaxis_title="Normalized Intensity",
        sliders=sliders,
    )
    html_filename = "data/interactive_fit_plots.html"
    fig.write_html(html_filename)
    fig.show()
    print(f"Interactive T₁ fit viewer saved as '{html_filename}'.")


def interactive_2D_contour(x, y, z):
    """
    Create an interactive Plotly 2D contour plot that mimics the static Matplotlib contour.

    The function creates two traces, one for positive data (blue lines) and one for negative data (red lines),
    computed using the same contour levels as the Matplotlib version.

    Parameters:
      x (array-like): F2 axis in ppm.
      y (array-like): F1 axis in Hz.
      z (2D np.ndarray): The spectral data.
    """
    # Compute contour levels as in Matplotlib.
    lv = np.geomspace(0.01 * np.max(z), np.max(z), num=10)
    contour_size = (lv[-1] - lv[0]) / (len(lv) - 1)

    trace_pos = go.Contour(
        x=x,
        y=y,
        z=z,
        contours=dict(start=lv[0], end=lv[-1], size=contour_size, coloring="lines"),
        line=dict(color="blue"),
        showscale=False,
    )

    trace_neg = go.Contour(
        x=x,
        y=y,
        z=-z,
        contours=dict(start=lv[0], end=lv[-1], size=contour_size, coloring="lines"),
        line=dict(color="red"),
        showscale=False,
    )

    fig = go.Figure(data=[trace_pos, trace_neg])
    fig.update_layout(
        title="Interactive 2D Contour Plot",
        xaxis_title="F2 (ppm)",
        yaxis_title="F1 (Hz)",
    )
    fig.update_xaxes(autorange="reversed")
    html_filename = "data/interactive_2D_contour.html"
    fig.write_html(html_filename)
    fig.show()
    print(f"Interactive 2D contour viewer saved as '{html_filename}'.")


def main():
    """
    Main routine for processing and analyzing 2D NMR data.

    This routine performs the following steps:
      1. Reads and processes raw data using the Proc2D module.
      2. Asks the user to select the desired F2 axis unit (Hz or ppm).
      3. Displays the 2D contour (static and interactive versions).
      4. Prompts the user for interactive peak selection.
      5. For each selected peak, computes integrals and performs exponential fitting.
         The user is prompted for the type of fit ('mono' or 'bi').
      6. Saves static plots and interactive HTML viewers for delay spectra, T₁ fits, and 2D contour plots.

    Author: Rajka Pejanovic, WCOLEN Normale Supérieure
    """

    folder = "4"
    procno = 2
    workdir = os.path.join(folder)
    os.makedirs("data_diff", exist_ok=True)
    os.makedirs("data/total_spectra_per_delay", exist_ok=True)

    # Instantiate and process data.
    proc2d = Proc2D(workdir, procnum=procno)
    proc2d.ReadFiles()
    proc2d.Proc()  # now proc2d.ser, proc2d.sw2scale (Hz), proc2d.swp2scale (ppm), etc. are available
    vdlist = np.array(proc2d.vdlist)

    # --- Choose F2 axis unit ---
    unit_choice = input("Enter F2 axis unit (Hz/ppm) [default: ppm]: ").strip().lower()
    if unit_choice == "hz":
        f2_axis = proc2d.sw2scale
    else:
        f2_axis = proc2d.swp2scale

    # --- Print F2 Axis information and generate 2D contour plot ---
    print("F2 axis range (Hz):", proc2d.sw2scale.min(), "to", proc2d.sw2scale.max())
    f1_axis = np.linspace(0, proc2d.sw1h / 2, proc2d.ser.shape[0])
    lv = np.geomspace(0.02 * np.max(proc2d.ser), np.max(proc2d.ser), num=10)
    X, Y = np.meshgrid(proc2d.swp2scale, f1_axis)
    plt.figure("2D Contour")
    plt.contour(X, Y, proc2d.ser, levels=lv, colors="b")
    plt.contour(X, Y, -proc2d.ser, levels=lv, colors="r")
    plt.gca().invert_xaxis()
    plt.xlabel("F2 (ppm)")
    plt.ylabel("F1 (Hz)")
    plt.title("2D Contour Plot")
    plt.tight_layout()
    plt.savefig("data/2D_contour.svg", dpi=300)
    plt.close()
    # Launch interactive 2D contour viewer (mimics static plot)
    interactive_2D_contour(proc2d.swp2scale, f1_axis, proc2d.ser)

    # --- Calculate F2 indices for specified ppm positions ---
    signal_ppms = [3.26, 3.94, 4.67, 7.3, 8.57]
    signal_Hz = [ppm * proc2d.bf1 for ppm in signal_ppms]  # convert to Hz
    f2_indices = [np.argmin(np.abs(proc2d.sw2scale - hz)) for hz in signal_Hz]
    print("Requested ppm positions:", signal_ppms)
    print("F2 indices used:", f2_indices)
    for idx in f2_indices:
        print(f"Max amplitude at F2 idx {idx}:", np.max(np.abs(proc2d.ser[:, idx])))
    print("Shape of proc2d.ser:", proc2d.ser.shape)

    # --- Static Peak Map and Interactive Peak Selection ---
    avg_spec = np.mean(proc2d.ser, axis=0)
    # plt.figure(figsize=(10, 5))
    # plt.plot(proc2d.swp2scale, avg_spec, label="Average Spectrum")

    interactive_regions = interactive_peak_selection(proc2d.swp2scale, avg_spec)

    peak_regions = []
    for reg in interactive_regions:
        idx_low = np.argmin(np.abs(proc2d.swp2scale - reg[0]))
        idx_high = np.argmin(np.abs(proc2d.swp2scale - reg[1]))
        start_idx, end_idx = sorted([idx_low, idx_high])
        peak_regions.append((start_idx, end_idx))
    plt.figure(figsize=(10, 5))
    plt.plot(proc2d.swp2scale, avg_spec, label="Average Spectrum")

    for i, (start, end) in enumerate(peak_regions):
        center = (start + end) // 2
        plt.axvspan(
            proc2d.swp2scale[start], proc2d.swp2scale[end], color="orange", alpha=0.3
        )
        plt.text(
            proc2d.swp2scale[center],
            max(avg_spec) * 0.8,
            f"Peak {i+1}",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=90,
        )
    plt.gca().invert_xaxis()
    plt.xlabel("F2 (ppm)")
    plt.ylabel("Intensity")
    plt.title("Average Spectrum with Selected Peak Regions")
    plt.tight_layout()
    plt.savefig("data/peak_map.svg", dpi=300)
    plt.close()

    # --- Prepare lists for fit processing ---
    report_lines = ["Peak\tIndex Range\tT1 (s)\t± Error\tM0\t± Error\tM_inf\t± Error"]
    all_reports = []
    all_fits = []  # each tuple: (vdlist, integrals, fitted_curve)
    all_t1_labels = []  # T1 values and uncertainties

    # --- Process Each Peak ---
    for i, (start, end) in enumerate(peak_regions):
        print(f"\nProcessing Peak {i+1}: indices {start}-{end}")
        region_spectra = proc2d.ser[:, start:end].copy()
        for j in range(region_spectra.shape[0]):
            region_spectra[j, :] = SimpleBaselineCorrection(region_spectra[j, :])
        # Save static stacked spectra.
        plt.figure(figsize=(8, 6))
        offset = 0.0
        for j, row in enumerate(region_spectra):
            plt.plot(
                proc2d.swp2scale[start:end],
                row + offset,
                label=f"VD {j+1}" if j == 0 else "",
            )
            offset += np.max(np.abs(region_spectra[0])) * 1.2
        plt.title(f"Stacked Spectra - Peak {i+1}")
        plt.xlabel("F2 (ppm)")
        plt.ylabel("Offset Intensity")
        plt.gca().invert_xaxis()
        plt.tight_layout()
        plt.savefig(f"data/stacked_peak_{i+1}.svg", dpi=300)
        plt.close()

        # Integration over the peak region.
        integrals = np.array([np.sum(row) for row in region_spectra])
        integrals = integrals / np.max(np.abs(integrals))

        # Create an ExponentialFitting instance for the current peak.
        fit = ExponentialFitting(vdlist, integrals)
        # Ask the user which fit type to use.
        fit_type = (
            input(f"Peak {i+1}: Enter fit type ('mono' or 'bi') [default: mono]: ")
            .strip()
            .lower()
        )
        if fit_type not in ["mono", "bi"]:
            fit_type = "mono"
        if fit_type == "mono":
            print("Using mono-exponential fitting.")
            fit.fit_mono()
        else:
            print("Using bi-exponential fitting.")
            fit.fit_bi()

        with open(f"data/report_peak_{i+1}.txt", "w") as f:
            f.write(fit.report)
        all_reports.append(f"--- Peak {i+1} ---\n" + fit.report)
        res = fit.fit_result.params
        T1_val, T1_err = res["T1"].value, res["T1"].stderr
        M0_val, M0_err = res["M0"].value, res["M0"].stderr
        Minf_val, Minf_err = res["M_inf"].value, res["M_inf"].stderr
        report_lines.append(
            f"{i+1}\t{start}-{end}\t{T1_val:.4f}\t±{T1_err:.4f}\t{M0_val:.4f}\t±{M0_err:.4f}\t{Minf_val:.4f}\t±{Minf_err:.4f}"
        )

        T1_val = fit.fit_result.params["T1"].value
        T1_err = fit.fit_result.params["T1"].stderr
        label_fit = f"Fit: T₁ = {T1_val:.2f} ± {T1_err:.2f} s"

        plt.figure()
        plt.title(f"T₁ Fit - Peak {i+1}")
        plt.xlabel("Inversion Delay (s)")
        plt.ylabel("Normalized Intensity")
        plt.scatter(vdlist, integrals, label="Data", alpha=0.6)
        plt.plot(
            vdlist,
            fit.MonoExpResidual(fit.fit_result.params, vdlist),
            label=label_fit,
            linewidth=2,
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"data/fit_peak_{i+1}.svg", dpi=300)
        plt.close()

        all_fits.append(
            (
                vdlist.copy(),
                integrals.copy(),
                fit.MonoExpResidual(fit.fit_result.params, vdlist),
            )
        )
        all_t1_labels.append(f"T₁ = {T1_val:.2f} ± {T1_err:.2f} s")

    with open("data/T1_summary.tsv", "w") as f:
        f.write("\n".join(report_lines))
    with open("data/full_report.txt", "w") as f:
        f.write("\n\n".join(all_reports))

    # Combined static plot of all fits.
    plt.figure(figsize=(10, 6))
    for i, (x, y, y_fit) in enumerate(all_fits):
        plt.scatter(x, y, label=f"Peak {i+1} data", alpha=0.6)
        plt.plot(x, y_fit, label=f"Peak {i+1} fit", linewidth=2)
    for i, label in enumerate(all_t1_labels):
        plt.text(
            0.05,
            0.95 - 0.07 * i,
            label,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(facecolor="white", edgecolor="black"),
        )
    plt.xlabel("Inversion Delay (s)")
    plt.ylabel("Normalized Intensity")
    plt.title("All T₁ Fits - Inversion Recovery Summary")
    plt.grid(True)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig("data/all_fits_combined.svg", dpi=300)
    plt.close()

    # Save full spectra per VD (static).
    for i, row in enumerate(proc2d.ser):
        plt.figure(figsize=(10, 4))
        plt.plot(proc2d.swp2scale, row, label=f"VD {i+1}")
        for j, (s, e) in enumerate(peak_regions):
            plt.axvspan(
                proc2d.swp2scale[s],
                proc2d.swp2scale[e],
                alpha=0.3,
                label=f"Peak {j+1}" if i == 0 else None,
            )
        plt.title(f"Spectrum for VD {i+1}")
        plt.xlabel("F2 (ppm)")
        plt.ylabel("Intensity")
        plt.legend()
        plt.gca().invert_xaxis()
        plt.tight_layout()
        plt.savefig(f"data/total_spectra_per_delay/spectrum_vd_{i+1}.svg", dpi=300)
        plt.close()

    print(
        "\n✅ All spectra processed and saved in 'data/' including T₁ summary, reports, and spectra per delay."
    )

    # --- Launch interactive viewers ---
    interactive_delay_viewer(proc2d.sw2scale, proc2d.ser, vdlist)
    interactive_fit_viewer(all_fits, all_t1_labels)


if __name__ == "__main__":
    main()
