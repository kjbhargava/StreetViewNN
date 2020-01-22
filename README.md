# StreetViewNN

This project needs the datasets from the following sites: http://ufldl.stanford.edu/housenumbers/ (format 2) using only the train and test data sets.  If you download this repo, please download the dataset associated with it as well and change the relative path on your files.

Description:  This project was meant to analyze the robustness of a Convolutional Neural Network's architecture.  It was trained on "perfect" images of streetview House numbers formatted to mimic the style of the MNIST data set.  The model was then tested on MNIST classic to see how well a CNN trained on "perfect" digits would be able to recognize "imperfectly" drawn digits.  Techniques implemented were edge detection, gray-scale, mean normalization, etc to help feature engineer and provide optimal results. 
