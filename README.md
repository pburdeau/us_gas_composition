# High-resolution national mapping of natural gas composition redefines methane leakage impacts

This repository contains the code and data processing pipeline used in the study:

**Burdeau, P.M., Sherwin, E.D., Biraud, S.C., Berman, E.S.F., Brandt, A.R.**  
*High-resolution national mapping of natural gas composition redefines methane leakage impacts*  
Preprint available at: [https://www.researchsquare.com/article/rs-6531662/v1](https://www.researchsquare.com/article/rs-6531662/v1)

## Overview

The study introduces a new method to generate high-resolution estimates of produced gas composition across the U.S., combining spatio-temporal kriging and a non-linear model based on oil and gas production data. The results improve methane loss rate estimates and enable more accurate emission inventories.

## Main Features

- Integration of data from:
  - USGS gas composition samples (1918–2014)
  - GHGRP production and processing data (2015–2021)
  - Enverus DrillingInfo well-level production data (1916–2022)
- Spatio-temporal kriging and neural network modeling
- Basin-level aggregation and uncertainty quantification
- Computation of methane loss rates with updated produced gas composition

## Repository Contents

- `src/`: Core code and functions
- `notebooks/`: Example notebooks and scripts
- `data/`: Processed data (when available)
- `output/`: Results and visualizations
- `README.md`: Project overview

## Citation

If you use this code or data, please cite the paper linked above.

## Contact

For questions or collaborations, contact:  
**Philippine Burdeau** – pburdeau@stanford.edu
