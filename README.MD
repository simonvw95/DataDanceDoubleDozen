# DataDance DoubleDozen

For more detailed explanations on the method and results, as well as the visualizations of the results, please visit the following page: https://simonvw95.github.io/DataDanceDoubleDozen/

# Environment instructions

Python version 3.10.4

Install packages with: 'pip install -r requirements.txt'

## Recreating results 

To recreate the DataDance DoubleDozen (For the DataSaurus Dozen, skip step 1 and 4):
1. Run **datadance_to_dataset.py**. 
- In this script, specify the number of points used to display the rickroll gif (and thus the number of points in your data sets). This by default is N=250.
2. Run **generate_shapes.py**.
- In this script, specify the number of points you want for each target data set. This by default is N=250.
- This script also produces a random cloud dataset with N points with the same summary statistics as the random cloud data set with 142 points.
- Each data set is stored in the target_datasets folder. You'll have to move the random_cloud_{N}.csv to the seed_datasets folder to use it as a starting data set.
3. Run **main.py**. This script contains multiple variables that can be adjusted by the user:
- it: the number of iterations used by the algorithm.
- de: the number of decimals that the summary statistics should remain the same
- frames: the number of images we want to return as intermediate results and therefore the number of frames in our progression gifs
- max_shake, min_shake: the maximum and minimum amount of noise we add to points, respectively.
- max_temp, min_temp: the maximum and minimum temperature of the algorithm, respectively. I.e. the probability of accepting bad data sets.
- max_sample_divis: the percentage of points we select in our subset to which noise is added.
- min_sample: the minimum sample size
- func_list: a list (with a minimum length of 'it') containing which distance function to use for which iteration. Set to \[Chamfer\] * (it * 0.9) + \[Hungarian\] * (it * 0.1) for the DataSaurus Dozen and the reversed order for the DataDance DoubleDozen
- shape_start: a string indicating which data set in seed_datasets to use as the initial data set
- shape_ends: a list of strings indicating which data set(s) in target_datasets to use as to replicate their shapes.
4. Run **datadance_gif.py**. In this script, specify the names of the png files to use to create the final gif of the DataDance DoubleDozen.
