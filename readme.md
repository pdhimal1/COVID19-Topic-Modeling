```
Prakash Dhimal
Manav Garkel
George Mason University
CS 657 Mining Massive Datasets
Assignment 3: Topic Modeling
```

#### Project organization:
  * `src` directory
    * `hw3.py` -> source code for homework 3
  * `report` directory
    * `report.pdf`
  * `out` directory
    * Few output files produced by this program
  * `readme.md`

###### `out` directory:
We utilize pyLDAvis, a python library for interactive topic model visualization to visualize the topics uncovered using LDA.
This tool extracts information from a fitted LDA topic model to inform an interactive web-based
visualization that can be saved to a stand-alone HTML file for portability.

The output directory `out/` contains several of these HTML file for easy visualization.
Full dataset run for 5, 7, and 10 topics are included in this directory.


#### Data:
This program depends on data in the `data/archive/` directory. The size of the data for this assignment
was simply too large (`>23GB`) to be included here. We recommend that the `archive` directory from Kaggle be placed
in `data/` directory to successfully run this program. You may change the `data_path = '../data/archive/'` line
[line 208] in [hw3.py](src/hw3.py).

#### Dependency:
This python program depends on the following modules:
  * time
  * pyspark
  * os
  * pyLDAvis - for visualization
  * nltk - for stopwords


#### How to run this program:
  * Navigate to the `src` directory and run one of the files using the command below:
  * `spark-submit hw3.py` OR `python hw3.py`


