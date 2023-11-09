[![UKIS](docs/ukis-logo.png)](https://www.dlr.de/eoc/en/desktopdefault.aspx/tabid-5413/10560_read-21914/) UKIS-PaperFairGreen
=======================================

This repository contains the scripts developed for the following research paper:

Weigand, M., Wurm, M., Droin, A., Stark, T., Staab, J., Rauh, J., Taubenböck, H. (under review). Are
public green spaces distributed fairly? A nationwide analysis based on remote sensing, OpenStreetMap
and census data. Submitted to Geocarto International. DOI WILL BE ADDED UPON PUBLISHING.

The code was developed at the [German Aerospace Center (DLR)](https://www.dlr.de)

# Contents

- `01_green_db` contains all scripts required to set up and fill a PostgreSQL/PostGIS database that
  is used for efficient data access. Scripts are ordered numerically.
- `02_green_extract` contains functionality to extract green space related information as part of
  preprocessing. A `.h5` array is created to facilitate efficient ingestion into the deep learning
  pipeline. The entry point is `h5extractor.py`.
- `03_fusion_green` contains the TensorFlow-based deep learning pipeline for training, evaluating
  and predicting public green space availability on neighborhood scale nationwide. The central entry
  point is `experiment.py` and the configuration used in the published paper is located in
  `runs/final_run/config.yml`.

# Licenses
This software is licensed under the [Apache 2.0 License](LICENSE.txt).

Copyright (c) 2023 German Aerospace Center (DLR) * German Remote Sensing Data Center * Department:
Geo-Risks and Civil Security

# Contributing
The UKIS team welcomes contributions from the community.  For more detailed information, see our
guide on [contributing](CONTRIBUTING.md) if you're interested in getting involved.

# What is UKIS?
The DLR project Environmental and Crisis Information System (the German abbreviation is UKIS, standing for [Umwelt- und Kriseninformationssysteme](https://www.dlr.de/eoc/en/desktopdefault.aspx/tabid-5413/10560_read-21914/) aims at harmonizing the development of information systems at the German Remote Sensing Data Center
(DFD) and setting up a framework of modularized and generalized software components.

UKIS is intended to ease and standardize the process of setting up specific information systems and thus bridging the gap from EO product generation and information fusion to the delivery of products and information to end users.

Furthermore, the intention is to save and broaden know-how that was and is invested and earned in the development of information systems and components in several ongoing and future DFD projects.
