To run:
python3 regression.py

This program uses polynomial regression of degrees 1, 2, 4, and 7 to model data. It achieves this by implementing full batch gradient descent on each polynomial term's weight with an alpha value of 0.001. The gradient descent stops after the largest weight update is less than 0.00001.

The program plots the result and prints the mean squared error of each regression.