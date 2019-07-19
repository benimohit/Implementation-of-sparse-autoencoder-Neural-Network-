
Need to implement sampleIMAGES() method.
Will use to load scipy.io.loadmat ‘IMAGES.mat’ file.
Then we will take the ‘IMAGE’ attribute which hold the 10 images. The training set will be obtained by randomly picking one of the 10 images and then randomly selecting an 8×8 image patch from the selected image, and by converting the image patch into a 64-dimensional vector. We will take 10000 samples and concatenated into a 64×10000 matrix.


