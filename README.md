# OccamSAM

A Python implementation of the OccamSAM algorithm, which is the first convex SLAM/SAM algorithm to automatically perform data association simultaneously while estimating the optimal location of robot and landmark positions. For more details, please refer to the following publications:

1. [Simultaneous Localization and Layout Model Selection in Manhattan Worlds](https://ieeexplore.ieee.org/document/8613887)
   - **NOTE:** The Manhattan World constraint is relaxed to a Stata-Center World constraint in this implementation
2. LINK TO NEW PAPER

Note that OccamSAM **assumes that robot orientations are known and observable**.

## Usage

### Creating new variables

The first step in using this package is to instantiate new `Variable` types from the `variable.py` module for each
robot keyframe and landmark encountered.

You can instantiate new robot and landmark position variables simply by defining its dimensionality:
```markdown
point1 = PointVariable(3)
point2 = PointVariable(3)
landmark = LandmarkVariable(1)
```

### Creating new factors

Smoothing-And-Mapping (SAM) algorithms perform and optimization over variables constrained by various types of factors. 

The `factor.py` module contains three different types of factors for defining constraints on variables
1. `PriorFactor` : Anchors a single `PointVariable` to a specified location, typically used for initialization. At least one of these factors must
    be provided in order to have a well-defined optimization problem. e.g., 
    ```markdown
    f = PriorFactor(point1, A, b)
    ```
2. `OdometryFactor` : The vector translation measurement between consecutive pairs of `PointVariable`s. e.g., 
    ```markdown
    f = OdometryFactor(point1, point2, R, t)
    ```
3. `ObservationFactor` : The vector range measurement to a `LandmarkVariable` from a `PointVariable`. e.g.,
    ```markdown
    f = ObservationFactor(point1, landmark, R, d)
    ```

Please consult the documentation within the `factor.py` module for information regarding the specific parameters required
to construct each factor. Additional Gaussian noise characteristics may be provided for each factor.


### Building a factor graph

Variables and the linear factors joining them are stored within a `GaussianFactorGraph` from the `factorgraph.py` module.

A factor graph can be instantiated as follows,
```markdown
fg = GaussianFactorGraph()
```

If you wish to only maintain a window of at most `n` `PointVariable`s for the optimization procedure to consider, you can specify a `free_point_window`.
```markdown
fg = GaussianFactorGraph(free_point_window=n)
```
In an online setting, doing so can help increase performance by bounding complexity. 

Factors (and the variables they contain) are added to the factor graph using the `add_factor()` method.

### Optimization

The different optimization strategies for the linear SAM problem can be found in the `optim.py` module. 
Two basic algorithms, `LeastSquares` and `WeightedLeastSquares`, are provided in addition to our main
`Occam` algorithm we showcase here. Additional custom algorithms may be implemented by specifying
an `optimize()` method for performing SAM inference and an `update()` method for updating the variable
positions.

```markdown
opt = Occam(fg)
opt.optimize()
opt.update()
```

## Dependencies

Python 2 and 3 compatible.

Required Packages:
- NumPy
- SciPy
- CvxPy

Optional Packages:
- Gurobi
- Mosek
- CvxOpt






