# Starlight
Data-driven, hierarchical models of Milky Way stars
Fluxes, colors, parallaxes, and population/hierarchical parameters

[![Build Status](https://travis-ci.org/ixkael/Starlight.svg?branch=master)](https://travis-ci.org/ixkael/Starlight)
[![Coverage Status](https://coveralls.io/repos/github/ixkael/Starlight/badge.svg?branch=master)](https://coveralls.io/github/ixkael/Starlight?branch=master)
[![Latest PDF](https://img.shields.io/badge/PDF-latest-orange.svg)](https://github.com/ixkael/Starlight/blob/master/paper/shrinkingparallaxes.pdf)


**Tests**: pytest for unit tests, PEP8 for code style, coveralls for test coverage.

## Content

**./paper/**: journal paper describing the method </br>
**./starlight/**: main code (Python/Cython) </br>
**./tests/**: test suite for the main code </br>
**./cluster_sources/**: Open clusters members </br>
**./notebooks/**: demo notebooks using starlight </br>
**./data/**: some useful inputs for tests/demos </br>

## Requirements

Python 3.5, cython, numpy, scipy, pytest, sfdmap, astropy, coveralls, matplotlib </br>

If you run the notebooks, you will need the 3D dust maps, as packaged in [dustmaps](http://dustmaps.readthedocs.io/en/latest/index.html).

## Authors

Boris Leistedt (NYU) </br>
David W. Hogg (NYU) (Flatiron)

## License

Copyright 2016-2017 the authors. The code in this repository is released under the open-source MIT License. See the file LICENSE for more details.
