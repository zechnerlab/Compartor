
# Compartor

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zechnerlab/Compartor/master)

Compartor is an automatic moment equation generator for stochastic compartment populations:
From a set of provided interaction rules, it derives the associated system of ODEs that describes the population statistics.
Equations can be exported as Python or Julia code.


## Installation

Install with `pip`
```
pip install compartor
```

Please that the LPAC-related notebooks use the local copy of compartor in the repository, as the pip package is (temporarily) still on version 1.1.1.


## Getting started

A tutorial for the usage of Compartor is provided as Jupyter notebooks at https://github.com/zechnerlab/Compartor.
(You can experiment with the notebooks on [Binder](https://mybinder.org/v2/gh/zechnerlab/Compartor/master) first, without installing anything locally.)

## Publications

Compartor is published in

Pietzsch, T., Duso, L., Zechner, C. (2020)
Compartor: a toolbox for the automatic generation of moment equations for dynamic compartment populations.
*Bioinformatics,* 2021, btab058, (https://doi.org/10.1093/bioinformatics/btab058).

Compartor is based on the theoretical framework introduced by

Duso, L. and Zechner, C. (2020).
Stochastic reaction networks in dynamic compartment populations.
*Proceedings of the National Academy of Sciences,* **117**(37), 22674–22683 (https://doi.org/10.1073/pnas.2003734117).


