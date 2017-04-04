# pseudofitter

"Pseudo fitter" is a Python-based spectral analysis routine written by Sara Camnasio. The fitting routine uses the Pyspeckit package.

The output plots above were made using the different functionalities:

deriv.py – calculates local minimum, local maximum and inflection points of a polynomial fit to a spectrum in terms of wavelength
newcoefficients.py – plots values of 5 polynomial coefficients across a table of spectral types and colors (with uncertainties)
pseudo_fitter.py – fits a polynomial of nth-degree to an array of data using Pyspeckit. This also outputs mean and standard deviation of the fit coefficients using Monte Carlo uncertainty analysis
sequence_plots.py – creates spectral sequence plots of three objects pulling local data files
