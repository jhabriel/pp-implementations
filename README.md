# A collection of numerical tests using PorePy

This GitHub repository is intended to serve as a (almost personal) collection of numerical tests for problems in saturated-unsaturated deformable-fractured porous media. All the implementations are part of the PhD project: Fracturing of porous media in the presence of multiphase flow at the University of Bergen, Norway. 

The test cases are presented in Jupyter Notebooks and are intended to be self-explanatories (at least to a certain degree).

To be able to run the notebooks, you should have a Python 3.6 distribution installed in your machine and the latest PorePy realease (see www.github.com/pmgbergen/porepy). It will be wise to check at least a few tutorials from PorePy, specially the MPFA, MPSA, Biot and Automatic Differentiation tutorials, which are used extensively.

## Table of contents

### Unsaturated flow in non-fractured non-deformable porous media (a.k.a. Richards' equation)

1. richards.ipynb: Pseudo-one dimensional water infiltration in a homogeneous soil column.
2. convergence_richards.ipynb: Convergene analysis on an unit square using a manufactured solution.

### Saturated flow in non-fractured deformable porous media (a.k.a. Biot's equations)

1. terzaghi.ipynb: Pseudo-one dimensional classical Terzaghi's consoladation problem.
2. mandel.ipynb: Classical Mandel's two-dimensional consolidation problem
3. biot_convergence: Convergnce analyisi on an unit square using a manufactured solution.

### Unsaturated flow in non-fractured deformable porous media (a.k.a. Unsaturated Biot's equations)

1. unsat_poro_conv_test_1.ipynb: Convergence analysis #1 on an unit square using a manufactured solution.
