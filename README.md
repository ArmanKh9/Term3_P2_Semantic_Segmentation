# Semantic Segmentation
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)

 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow.
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy.
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.

### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well.

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.


# Write up
In this project, a fully convolutional neural network (FCN) is being trained and used to classify each pixel of an image as road or not road and mark it in green.

## Pre-Trained VGG model
A pre-trained VGG-16 model (frozen model) has been used to expedite the process of developing encoder part of the FCN model. The VGG-16 model used here is specifically trained on Kitty dataset. The VGG-16 model returns a 1x1 convoluted tensor in lieu of a fully connected layer. This preserves the spatial information that is necessary for upscaling, and in general, decoding each image.

## Decoder
The encoder part of the FCN takes in the VGG-16 output and upscale it to an image with the same size as the input image through multiple layers. It also perform skip layer operation on two layers in order to better detect edges of the road. In the skip layer operation, output of some layers in the encoder part are extracted and passed to the decoder section to be added to a tensor with the same size. However, these extracted layers need to be reshaped in order to have the same depth as the number of classes.

Decoder section:

```
#1x1 conv - This reduces the number of filters down to number of classes (#filters=#classes)
vgg_layer7_2class = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding = 'same',
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

#upsample
layer7_transpose = tf.layers.conv2d_transpose(vgg_layer7_2class, num_classes, 4, strides=(2,2), padding='same',
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))


#1x1 conv - This reduces the number of filters down to number of classes (#filters=#classes)
vgg_layer4_2class = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding = 'same',
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

#skip layer - adding convoluted form of output of layer 4 in the ecnoder to output of a layer in decoder with the same size
skip_layer4 = tf.add(layer7_transpose, vgg_layer4_2class)

#upsample
layer4_transpose = tf.layers.conv2d_transpose(skip_layer4, num_classes, 4, strides=(2,2), padding='same',
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

#1x1 conv - This reduces the number of filters down to number of classes (#filters=#classes)
vgg_layer3_2class = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding = 'same',
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))



#skip layer - adding convoluted form of output of layer 3 in the ecnoder to output of a layer in decoder with the same size
skip_layer3 = tf.add(layer4_transpose, vgg_layer3_2class)

#upsample
layer3_transpose = tf.layers.conv2d_transpose(skip_layer3, num_classes, 16, strides=(8,8), padding='same',
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
return layer3_transpose
```
## Training
Once developing the architecture of the model was completed, training process was initiated on Kitty dataset. In this training, the FCN model predicts class of each pixel as road or not road.

During the training process, images from the Kitty dataset were fed into the FCN model. The model uses cross entropy as a loss function in order to adjust the model weights (only in decoder) with learning rate of 1e-4. In order to avoid over-fitting, l2_regularizer function was implemented in each decoder layer. Batch size of 6 with 40 epochs was found to be adequate in training the model. The model provides a good prediction with 20 epochs but the loss function stabilized at around epoch 40.

I used Amazon Web Services to train this model. A GPU instance was initiated per instruction in the lessons and trained the model in a fairly short time.

```
sess.run(tf.global_variables_initializer())

    print("Training...")
    for epoch in range(epochs):
        print("Epoch ", epoch+1)
        for image, label in get_batches_fn(batch_size):
            #Training
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: image, correct_label: label,
                                                                keep_prob: 0.5, learning_rate: 0.0001})
            print("Loss: = {:.3f}".format(loss))
```

## Results

The FCN model showed a good performance in classifying pixels.

![good](https://github.com/ArmanKh9/Term3_P2_Semantic_Segmentation/blob/master/runs/1525048855.157215/um_000017.png)

However, the model is sensitive to contract and color change. In cases where there is a high contrast in light on the road, the model does not detect all road pixels.

![poor - contrast](https://github.com/ArmanKh9/Term3_P2_Semantic_Segmentation/blob/master/runs/1525048855.157215/um_000070.png)

Also in some instances, the model classifies non-road regions as road which can be due to the same RGB values in road and the non-road region.

![poor - RGB value](https://github.com/ArmanKh9/Term3_P2_Semantic_Segmentation/blob/master/runs/1525048855.157215/um_000067.png)
