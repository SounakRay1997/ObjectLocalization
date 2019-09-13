# ObjectLocalization

We have used a deep learning based approach to predict the bounding boxes of objects in the given images. As described in the problem statement, that this is a medium-complexity dataset, we thought that a training a simple three layer convolutional neural network in a end-to-end fashin will do the trick. This will save us the pain of arranging for a GPU to train our models and it will also give pretty good results.

At first we split the dataset into the train and test dataset based on the provided csv files. From the training dataset we randomly choose 40 images to serve as our validation set. The main challenge while preparing the code for our main model was the preparation of the training set. We wrote a custom data generator function which will pre-process the image  and prepare the batches for our training. The training images were resized to have a size of 224x224 pixels. This greatly reduced the number of parameters in our model. We choose our target vector to be the 4-dimensional vector, namely the vector (x1, x2, y1, y2) and trained our network with the mean-squared error (MSE) loss function. We also used early stopping and model checkpoint to reduce overfitting of our model. The whole is written in Keras. We also used learning rate reduction to obtain better performance. Our main model  consisted of Conv2D layers, relu activation layers and max pooling layers. The Conv2D and max pooling layers have been proven to be exceptionally powerful tool in capturing spatial correlation among image pixels and so we decided to use these as the basic building blocks. The efficient feature extraction is guaranted by the use of these layers. The last layer encodes the image into a single dimension with 4 channels. These 4 channels are our target vector. This is how we approached the FlipkartGRID Level 2 problem 
