# ML_scheduling
Data for paper Machine Learning Driven Many-objective Moving Horizon Scheduling Optimization
````markdown
# ML Scheduling: Machine Learning Driven Many-Objective Moving Horizon Scheduling Optimization

This repository contains the data, code, and plotting files associated with the paper:

**Machine Learning Driven Many-objective Moving Horizon Scheduling Optimization**  
Hongxuan Wang and Andrew Allman

The repository supports reproducibility of the machine-learning and optimization results reported in the manuscript. The study develops a machine-learning-enhanced moving horizon scheduling framework for electrified chemical processes. The framework predicts whether cost and carbon-emission objectives are correlated or competing from 48-hour electricity price and grid-emission-intensity profiles. These predictions are then used to decide whether a simplified grouped-objective scheduling solve is sufficient or whether a cost-emission tradeoff analysis is needed.

## Overview

Industrial electrification can reduce fossil-fuel use in chemical manufacturing, but it also exposes processes to time-varying electricity prices and grid carbon intensities. In many-objective moving horizon scheduling, cost, emissions, water usage, safety, and other sustainability objectives may be aligned under some grid conditions and conflicting under others.

This repository contains the datasets and scripts used to:

- Generate objective-relationship labels from many-objective scheduling problems.
- Train machine-learning classifiers to predict objective groupings from grid data.
- Compare random forest and LSTM models for objective-relationship prediction.
- Evaluate machine-learning-guided moving horizon scheduling for ammonia production.
- Compare cost-emission relationships between ammonia production and chlor-alkali electrolysis.
- Study transfer learning from ammonia scheduling to chlor-alkali scheduling.
- Reproduce the main figures and numerical results reported in the manuscript.


## Case Studies

### Ammonia Production

The ammonia case study considers a flexible ammonia production system with grid-powered hydrogen production, nitrogen generation, gas storage, and downstream ammonia synthesis. The scheduling model includes four objectives:

1. Operating cost
2. Carbon emissions
3. Water usage
4. Safety risk associated with electrolyzer startups

The ammonia dataset contains 48-hour electricity price and grid-emission-intensity windows with objective-grouping labels generated from the objective dimensionality reduction algorithm.

### Chlor-Alkali Electrolysis

The chlor-alkali case study is used to test whether objective-relationship prediction generalizes across electrified chemical processes. The simplified chlor-alkali scheduling model represents electricity-intensive electrolysis with current as the main operating decision. The primary objectives are:

1. Operating cost
2. Grid-related carbon emissions

The chlor-alkali dataset uses the same 48-hour grid input structure as the ammonia dataset, but the objective labels are generated from the chlor-alkali scheduling model. This allows the repository to support cross-process comparison and transfer-learning experiments.

## Machine Learning Tasks

The repository supports two main classification tasks.

### Five-Class Objective Grouping

The five-class task predicts the full objective grouping returned by the dimensionality reduction algorithm for the ammonia scheduling problem.

### Binary Cost-Emission Relationship Classification

The binary task predicts whether the cost and carbon-emission objectives are:

* **Correlating**: cost and emissions are assigned to the same objective group.
* **Competing**: cost and emissions are assigned to different objective groups.

The binary classification task is used in the online moving horizon scheduling workflow to decide whether to use a simplified solve or generate a cost-emission tradeoff frontier.

## Models

The manuscript evaluates both sequence-based and feature-based models.

### LSTM Classifier

The LSTM model uses the full 48-hour electricity price and grid-emission-intensity trajectories as a two-channel time series input.

Input format:

```text
N samples × 48 time steps × 2 channels
```

where the two channels are:

1. Electricity price
2. Grid carbon intensity

### Random Forest Classifier

The random forest model uses engineered statistical features extracted from the 48-hour price and emission trajectories, including summary statistics such as extrema, mean values, variance, and covariance.

## Optimization and Scheduling Workflow

The moving horizon scheduling workflow proceeds as follows:

1. Load a 48-hour electricity price and grid-emission-intensity window.
2. Predict whether cost and emissions are correlating or competing.
3. If objectives are correlating, solve a simplified grouped-objective scheduling problem.
4. If objectives are competing, generate a cost-emission tradeoff frontier and select a compromise solution.
5. Implement the first scheduling decision.
6. Shift the horizon forward and repeat.

This strategy avoids repeatedly running the full objective dimensionality reduction algorithm online while preserving tradeoff information when it is needed.

## Software Requirements

The code was developed using Python-based optimization and machine-learning workflows. The main software dependencies include:

* Python
* NumPy
* pandas
* scikit-learn
* TensorFlow
* matplotlib
* Pyomo
* IBM ILOG CPLEX Optimization Studio

Some analysis files may also use Julia.

Because CPLEX is a commercial solver, users must install CPLEX separately and ensure that the CPLEX executable or Python API is available in their local environment.

## Installation

Clone the repository:

```bash
git clone https://github.com/beam4869/ML_scheduling.git
cd ML_scheduling
```

Create and activate a Python environment:

```bash
conda create -n ml_scheduling python=3.10
conda activate ml_scheduling
```

Install common Python dependencies:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib pyomo
```

Install and configure CPLEX following IBM's instructions for your operating system.

## Data Description

The datasets contain processed 48-hour windows of electricity price and grid carbon intensity. Each window is paired with objective-relationship labels generated by solving the corresponding process scheduling problem and applying the objective dimensionality reduction algorithm.

The ammonia dataset supports:

* Five-class objective grouping classification
* Binary cost-emission relationship classification
* Moving horizon scheduling analysis
* Out-of-sample testing on ISO New England data

The chlor-alkali dataset supports:

* Binary cost-emission relationship classification
* Cross-process comparison with ammonia production
* Transfer learning from ammonia to chlor-alkali

## Reproducing Figures

The repository includes files for reproducing the main numerical figures, including:

* PCA projection and PCA-component classification analysis
* Machine-learning classification performance
* Moving horizon scheduling profiles
* Cost-emission Pareto frontier comparisons
* Tradeoff-length analysis
* Transfer-learning performance curves

Figure-specific files are organized in the corresponding analysis folders.



## Contact

For questions about the data, code, or manuscript, please contact:

**Hongxuan Wang**
University of Michigan
Email: [hongxuan@umich.edu](mailto:hongxuan@umich.edu)

```
```
