#!/usr/bin/python
"""
proc2D.py - A module for processing 2D Bruker NMR data.

This module provides the Proc2D class that reads the raw Bruker files
(acqus, acqu2s, ser, pdata, and vdlist), performs necessary processing
(FFT, digital filter removal, phasing, normalization, etc.), and generates
frequency axes both in Hertz (Hz) and in parts-per-million (ppm).

Example usage:
    from proc2D import Proc2D
    proc = Proc2D(workdir, procnum=2)
    proc.ReadFiles()
    proc.Proc()
    # Now the processed data is available as:
    #   proc.ser           -> Processed spectral data (2D array)
    #   proc.sw2scale      -> F2 frequency axis in Hz
    #   proc.swp2scale     -> F2 frequency axis in ppm
    #   proc.vdlist        -> List of delays (from vdlist)

Author: Rajka Pejanovic
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.fftpack import dct, idct


class Proc2D:
    """
    A class for processing 2D Bruker NMR data.

    The processing involves:
      - Reading Bruker parameter and data files.
      - Removing the digital filter.
      - Constructing the complex FID data.
      - Applying FFT in the F2 dimension with proper phasing.
      - Normalization and optional Lorentzian broadening.
      - Creating F2 frequency scales in both Hz and ppm.
    """

    def __init__(self, workdir, np1=0, np2=16400, procnum=2):
        """
        Initialize the Proc2D object.

        Parameters:
          workdir (str): Path to the working directory containing the Bruker files.
          np1 (int): The number of points in the first dimension (optional).
          np2 (int): The number of points in the second dimension (optional).
          procnum (int): The processing number (i.e., folder index in pdata) to use.
        """
        self.workdir = workdir
        self.np1 = int(np1)
        self.np2 = int(np2)
        self.procnum = procnum

    def RemoveDF(self):
        """
        Remove the digital filter from the raw data.

        This method performs an FFT along the F2 dimension of the raw data
        (self.serraw), applies n[] correction, and finally inverts the FFT
        to obtain a properly filtered spectrum stored in self.ser.

        The number of points to skip (skip) is computed from the group delay
        (self.grpdly).
        """
        # Apply FFT and shift data.
        self.ser = np.fft.fftshift(np.fft.fft(self.serraw, axis=1), axes=(1,))
        l = np.float32(self.serraw.shape[1])
        del self.serraw  # Free memory for raw data.
        # Create phase correction vector.
        phasecor = np.exp(1j * (np.arange(l) / l) * 2 * np.pi * self.grpdly)
        self.ser = self.ser * phasecor
        # Inverse FFT to return to time domain.
        self.ser = np.fft.ifft(np.fft.ifftshift(self.ser, axes=(1,)), axis=1)
        # Calculate the number of points to skip.
        skip = int(np.floor(self.grpdly) + 2.0)
        l = int(l)
        self.ser = self.ser[:, 0 : l - skip]
        print("Removed digital filter")

    def ReadFiles(self):
        """
        Read the necessary Bruker parameter and data files.

        Reads:
          - acqus: To obtain parameters such as TD, O1, BF1, SW_h, SW, GRPDLY, NC, DIGTYP.
          - ser: The raw free induction decay (FID) data (data type depends on DIGTYP).
          - acqu2s: To obtain first-dimension parameters such as TD, SW_h, SW, and FnMODE.
          - pdata/[procnum]/procs: To extract phasing parameters (PHC0, PHC1), SI, and WDW.
          - pdata/1/proc: To extract additional processing parameters (LB).

        The parameters are stored as attributes for later processing.
        """
        # Read "acqus" file.
        with open(self.workdir + "/acqus", "rt") as acqus:
            for line in acqus:
                words = line.split()
                if words[0] == "##$TD=":
                    self.td2 = np.int32(words[1])
                if words[0] == "##$O1=":
                    self.o1 = np.float32(words[1])
                if words[0] == "##$BF1=":
                    self.bf1 = np.float32(words[1])
                if words[0] == "##$SW_h=":
                    self.sw2h = np.float32(words[1])
                if words[0] == "##$SW=":
                    self.sw2 = np.float32(words[1])
                if words[0] == "##$GRPDLY=":
                    self.grpdly = np.float32(words[1])
                if words[0] == "##$NC=":
                    self.NC = np.float32(words[1])
                if words[0] == "##$DIGTYP=":
                    self.DIGTYP = np.int32(words[1])

        # Read the raw data from "ser" based on DIGTYP.
        if self.DIGTYP == 17:
            self.serraw = np.fromfile(self.workdir + "/ser", dtype=np.float64)
            self.serraw = np.float64(self.serraw)
        else:
            self.serraw = np.fromfile(self.workdir + "/ser", dtype=np.int32)
            self.serraw = np.float32(self.serraw)

        # Read "acqu2s" file.
        with open(self.workdir + "/acqu2s", "rt") as acqu2s:
            for line in acqu2s:
                words = line.split()
                if words[0] == "##$TD=":
                    self.td1 = np.int32(words[1])
                if words[0] == "##$SW_h=":
                    self.sw1h = np.float32(words[1])
                if words[0] == "##$SW=":
                    self.sw1 = np.float32(words[1])
                if words[0] == "##$FnMODE=":
                    self.fnmode = np.int32(words[1])

        # Read processing parameters from pdata.
        with open(
            self.workdir + "/pdata/" + str(self.procnum) + "/procs", "rt"
        ) as procs:
            for line in procs:
                words = line.split()
                if words[0] == "##$PHC0=":
                    self.phc0 = np.float32(words[1])
                    print("Phase 0", self.phc0)
                if words[0] == "##$PHC1=":
                    self.phc1 = np.float32(words[1])
                if words[0] == "##$SI=":
                    self.si2 = np.int32(words[1])
                if words[0] == "##$WDW=":
                    self.wdw = np.int32(words[1])

        with open(self.workdir + "/pdata/1/proc", "rt") as proc:
            for line in proc:
                words = line.split()
                if words[0] == "##$LB=":
                    self.lb = np.float32(words[1])

        print("Read files.")
        print("Number of points:", self.td1, "x", self.td2)
        print("Spectral widths:", self.sw1h, "[Hz] x", self.sw2h, "[Hz]")
        print("Fnmode:", self.fnmode)

    def Readvdlist(self):
        """
        Read the vdlist file.

        This file contains the delay values (one per line) for the indirect dimension.
        The number of delays is expected to equal td1.
        """
        with open(self.workdir + "/vdlist", "rt") as vdlist:
            lines = vdlist.readlines()
        n = len(lines)
        if n != self.td1:
            print("WARNING!")
        self.vdlist = [np.float32(line.split()[0]) for line in lines]

    def Proc(self):
        """
        Process raw data into a 2D spectrum and generate frequency axes.

        Steps:
          1. Reshape the raw data to a 2D array of size [td1, td2].
          2. Combine successive pairs of numbers to create complex FID data.
          3. Remove the digital filter using RemoveDF().
          4. Apply normalization (e.g., multiply the first point by 0.5 and multiply by 2^(2*NC)).
          5. Optionally apply Lorentzian broadening if WDW == 1.
          6. Perform an FFT along the F2 dimension and apply phase correction.
          7. Flip the spectral data so that the F2 axis is in the correct order.
          8. Construct the F2 frequency axis in Hz based on O1 and SW_h.
          9. Construct the F2 axis in ppm by converting O1 to ppm using BF1.
          10. Read the vdlist.

        After processing, the following attributes are available:
          - ser: The processed 2D spectrum (real-valued).
          - sw2scale: F2 axis in Hz.
          - swp2scale: F2 axis in ppm.
          - vdlist: List of delay values.
        """
        # Reshape raw data.
        self.serraw = self.serraw.reshape(-1, self.td2)
        self.serraw = self.serraw[0 : self.td1, :]
        # Create complex data from interleaved real and imaginary parts.
        self.serraw = self.serraw[:, 0::2] + 1j * self.serraw[:, 1::2]
        # Remove the digital filter.
        self.RemoveDF()
        print("First FID points multiplied by 0.5")
        self.ser[:, 0] = 0.5 * self.ser[:, 0]
        print("Normalization by *2^NC: ", self.NC)
        self.ser = self.ser * np.power(2, 2 * self.NC)
        # Optionally apply Lorentzian broadening.
        if self.wdw == 1:
            print("Applying Lorentzian broadening with ", self.lb, " Hz")
            LorBroadening = np.exp(
                -(np.arange(self.ser.shape[1])) * 2 * np.pi * self.lb / (2 * self.sw2h)
            )
            self.ser = self.ser * LorBroadening

        print("FFT in F2 and phasing")
        npts = self.si2
        if self.ser.shape[1] < npts:
            pad_width = npts - self.ser.shape[1]
            self.ser = np.pad(self.ser, ((0, 0), (0, pad_width)), mode="constant")

        self.ser = np.fft.fftshift(np.fft.fft(self.ser, axis=1, n=self.si2), 1)
        # Apply phase corrections.
        phase0cor = np.exp(-1j * 2 * np.pi * self.phc0 / 360)
        l = len(self.ser[0, :])
        phase1cor = np.exp(-1j * (np.arange(l) / l) * 2 * np.pi * self.phc1 / 360)
        self.ser = self.ser * phase0cor * phase1cor
        self.ser = np.real(self.ser)
        # Flip the data so that the F2 axis runs in the correct order.
        self.ser = np.flip(self.ser, axis=1)

        # Construct the F2 axis in Hz using the reference O1 and spectral width SW_h.
        sw2scale = np.linspace(
            self.o1 + self.sw2h / 2, self.o1 - self.sw2h / 2, self.ser.shape[1]
        )
        # Construct the F2 axis in ppm.
        # Convert the reference frequency to ppm.
        self.o1p = self.o1 / self.bf1
        swp2scale = np.linspace(
            self.o1p + self.sw2h / (2 * self.bf1),
            self.o1p - self.sw2h / (2 * self.bf1),
            self.ser.shape[1],
        )
        self.sw2scale = sw2scale  # F2 axis in Hz.
        self.swp2scale = swp2scale  # F2 axis in ppm.

        # Read the delay list.
        self.Readvdlist()
