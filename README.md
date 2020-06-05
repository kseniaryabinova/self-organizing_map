# Self-organized Kohonen map

Self-organized Kohonen map is a self-learning neural network. It is used for visualization and clusterization purposes.

This kind of neural network is used for dimensionality reduction (for example, from n-dimensional to 2-dimensional).   

## What?

Here is a Self-organized Kohonen map implemented on C++ with CUDA and OpenMP.

Also I used Python with matplotlib for visualization.

All of this was implemented in Microsoft Visual Studio.

## Why?

I had to implement some algorithm using parallel programing libraries for one of the subjects in my university. I chose to implement this neural network using CUDA and OpenMP and then to compare speed of this 2 implementations.

## Where are the results?

Here is the 100x100 map:

[100x100 map](images/map1.png)

Here a 3-dimensional vector space (space of an RGB vectors) is projected onto a 2-dimensional vector space (space of coordinates).

Learning is performed using rainbow colors: red, orange, yellow, green, light-blue, blue and purple.

Here is time-dimension plot: 
- X-axis - map size (from 100x100 to 1000x1000)
- Y-axis - seconds

[Time plot](images/??????.png)
