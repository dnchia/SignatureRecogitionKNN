Signature recognition
--
A signature classifier is implemented in this repository, using Tensorflow, as a part of a subject project in my master degree.

#### The employed process

The images are normalized before being processed, following these steps:
1) Rescale the images to 250x250
2) Convert the images to a gray-scale
3) Get the percentage of occupation in the signature (% of black)
4) Get the vertical and horizontal projections (% of black per column and row)
5) Get the grid values. In 10x10 grids, the minimum color value is taken, as the representative of the grid.
If there are any part of the signature trace, the black will be taken, otherwise, the white.
6) A vector of characteristics is created with all this data.

K-nearest Neighbors is used as a method to train the classifier.

#### The data

The signatures folder is not included in this project.
The used signatures folder has the following structure:

    pXX
       pXXsYY.png
        
There are 58 classes (XX) and 20 signatures per class (YY).

The signatures images have not the same size and the same color.

Once executed the code, the folders
- signatures_normalized
- signatures_normalized_model

are generated. The first one contains the preprocessed signatures, the second one, the preprocessed signatures in csv 
files, that are used to train the model.