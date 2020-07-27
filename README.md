# Different_CNN_on_MINST


## Introdcution:
In our project, we would like to compare different CNN architectures such as ALEXNET, VGGNet, and GOOGLENET on MINST data set, a large database of hand- written digits. After trying each of the architectures, we would compare each of their results based on training set size, test error, and efficiency. We would then perform data augmentation to compare based on robustness. In addition, we will observe the properties of each architecture that lead it to its results. Finally, we will try to implement our own program to achieve a similar result. CNNs are used today for many applications such as facial recognition and document analysis.

## Related Work

### AlexNet

On the ImageNet LSVRC-2010, in the classification task of a total of 1.2 million high-resolution images containing 1000 categories, AlexNet’s top-1 and top-5 error rates on the test set were 37.5% and 17.0%. AlexNet has 600 million parameters and 650,000 neurons, including 5 convolutional layers, some layers are followed by a max-pooling layer, and 3 fully connected layers. In order to reduce overfitting, dropout is used in the fully connected layer, and Use ReLU function as activation function. The main method it used was the Non-linear ReLU function. At that time, the standard neuron activation function was the tanh() function. Therefore, using the ReLU function in a 4-layer convolutional network as the activation function in AlexNet to achieve a training error rate of 25% on the CIFAR-10 dataset is 6 times faster than using the tanh function under the same network and the same conditions. Moreover,
Local Response Normalization was being used to reduce AlexNet’s top-1 and top-5 error rates by 1.4% and 1.2%, respectively and the Overlapping Pooling scheme reduces the top-1 and top-5 error rates by 0.4% and 0.3%. Last but not least, to prevent overfitting in the algorithm, dropout and data augmentation were the two main methods that were being used.

### VGG

The main work of the network is to prove that increasing the depth of the network can affect the final performance of the network to a certain extent. VGG has two structures, namely VGG16 and VGG19. There is no essential difference between the two, but the network depth is different. An improvement of VGG16 compared to AlexNet is to use several consecutive 3x3 convolution kernels to replace the larger convolution kernels in AlexNet (11x11, 7x7, 5x5). For a given receptive field, the use of stacked small convolution kernels is better than the use of large convolution kernels, because multiple nonlinear layers can increase the
depth of the network to ensure that learning is more complicated Mode, and the cost is relatively small. While VGG consumes more computing resources and uses more parameters, resulting in more memory usage. Most of the parameters are from the first fully connected layer. In conclusion, the author found through the network A and A-LRN that the local response normalization (LRN) layer used by AlexNet had no performance improvement. Moreover, with the increase of depth, the classification performance will also get
better. Lastly, the author found that multiple small convolution kernels have better performance than single large convolution kernels. The author did an experiment to compare B with one of his shallower networks not in the experimental group. The shallower network uses conv5x5 instead of B’s two conv3x3, and multiple small convolution kernels perform better than one single large convolution kernel. 

### GoogleNet

GoogleNet Structures such as AlexNet and VGG all obtain better training results by increasing the depth (number
of layers) of the network, but increasing the number of layers could also bring many negative effects, such as overfit, gradient disappearance, gradient explosion, etc. Inception is proposed to improve training results from another perspective: it can use computing resources more efficiently, and more features can be extracted under the same amount of calculation, thereby improving training results.

## Evaluation Method
The data set we are using is called MNIST (Mixed National Institute of Standards and Technology database), which is a large database of handwritten numbers collected by the National Institute of Standards and Technology, containing a training set of 60,000 examples and a test set of 10,000 examples. For more than ten years, researchers from the fields of machine learning, machine vision, artificial intelligence, and deep learning have used this data set as one of the benchmarks for measuring algorithms. It can be found in many conferences and journal papers. In fact, the MNIST data set has become one of the must-have data sets for algorithm authors. Someone once ridiculed: ”If an algorithm does not work in MNIST, then it will not work at all; and if it works on MNIST, it may not work on other data.” 

Before the training process starts, we do need to do some data augmentation to the existing data set, which includes rotation, noise adding and so on. In any case, the neural network will treat these slightly adjusted images as different images. Convolutional neural networks can classify objects robustly, even if the objects are placed in different directions, which is called the invariant property. More specifically, the convolutional neural network can stably respond to changes in image translation, image viewing angle, image size, or lighting conditions. This is actually the logical premise of image enhancement. In real work, we may get a data set in which the images are taken in some cases, but the target application is in many cases, such as different directions, positions, sizes, brightness, etc. We will train the neural network with additional revised data to deal with these situations.

After the training, we will evaluate the programing by comparing not only the prediction accuracy and loss, but also the performance of the Network such as time efficiency and if the model is overfitting. The specific performance of overfitting is as follows: the model has a small loss function on the training data and a high prediction accuracy; but the test function has a large loss function and the prediction accuracy is low. Therefore, the overfitting models are not usable. By applying or not applying the dropout and data augmentation, we will find different results, and we will summarize them and find the best among them.


## Reslult:

### ALexnet
<p align="center">
	<img src="https://github.com/bochendong/Different_CNN_on_MINST/raw/master/image/alex_1.png"
        width="1500" height="600">
	<p align="center">
</p>
<p align="center">
	<img src="https://github.com/bochendong/Different_CNN_on_MINST/raw/master/image/alex_2.png"
        width="1500" height="600">
	<p align="center">
</p>

