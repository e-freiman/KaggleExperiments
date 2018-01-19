# KaggleFacebookV

The goal of this repository is to lean ML through competition on Kaggle. For details of the competition look [here](https://www.kaggle.com/c/facebook-v-predicting-check-ins#description). The idea of the solution is straightforward implementation of Naive Bayes approach:

**p(x, y, accuracy, time | id) ~  p(x, y, accuracy, time | id) * p(id)**

**= p(id) * p(x, y) * p(accuracy) * p(time)**

**= p(id) * p(x, y) * p(accuracy) * p(day of the week) * p(time of the day)**

For modeling p(x, y) I used normal distributions, for modeling other probabilities I used histograms. In order to decrease amount of calculations, the solution has two phases: at first the parameters of the model are being calculated (gaussians for coordinates and histograms for time and accuracy), and being saved into files (small example is in *generated_model100* directory). This is done by Python script in Jupyter Notebook, look at *jupyter* directory. After that this model is being applied to test data. There are two implementations of that: in Python and C++ in according directories.

In order to generate files with the model go to *jupyter* directory and run *histograms.ipynb* and *gaussians.ipynb*, not forgetting to put proper filenames for train and output data manually. It'll take about 3-4 hours each. 

Since there is a large amount of the test data, Python implementation is too slow (about 270 hours at 20 cores machine on Windows Azure). This is after I vectorized all operations, made a c++ shared library for normal distribution computation and parallelized for many cores. 

Before running Python implementation you need to compile the shared library that is stored in *python/cpp* directory. There are *compile.bat* and *compile.sh* scripts for compiling it on both platforms. Since python is slow, it's better to test it on small subset of the real data, which is stored in the repository in *input* folder. The model for this small subset is already generated and also stored in repository in folder *generated_model100*. It contains only 100 places. To start testing, run facebook_v.py and specify model directory, test file and how often to print info about time that left.

C++ implementation works much faster (about 6 hours at 20 cores machine on Windows Azure). The code uses OpenMP for parallelization and was tested on two compilers: Visual C++ on Windows and g++ on Linux. The CMakeLists.txt file will help you to compile it smoothly on your compiler. After that run FacebookV.exe (if you use Windows) and specify  model directory, test file and 1/0 if the test data has answers or not (validation data in input folder has true labels, real test data from Kaggle hasn't). 

However this approach (Naive Bayes with normal distribution over locations) isn't good at all. The score I got is only 0.28. At the same time, [a much easier approach](https://www.kaggle.com/zfturbo/facebook-v-predicting-check-ins/mad-scripts-battle), that is just 110 lines long Python script that uses only coordinates got 0.45 (the winner got 0.62). 

Nonetheless, my goal was reached, I studied one of approaches of ML as well as efficient Python and C++ implementations, and I guess this repository is closed
