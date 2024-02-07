# Visiongebra

A Computer Vision library for Python, in Python.

## Project Details

This is a computer vision library that is completely built on
Vectorgebra. The development still continues, content is not
ready for use yet.

Requires a minimum of Vectorgebra 3.0.1.

## Image

Image class is the holder for any image data. An image is 
generated with "imread()" function. This function does not
belong to this class, but in most cases returns one. Some
error cases make the function return None. Only supported
image file type is currently PNG. And this format is not
fully supported. This function may return None for some
PNG images. Other image types are not supported, they
will raise FormatError. 

An Image instance, firstly holds the image Tensor. An image
Tensor is a collection of matrices that compose the image
itself. If we think of those matrices as layers, each layer
represents a color. Red, green and blue each gets represented
by another matrix, which in return form the general image
Tensor.

Other information that the Image object holds are; the load type,
the image type and the loaded color information. Loaded color
information may be different than the load type. 

Printing an Image will result in the inner Tensor getting printed.

Describing an Image will print out an explanatory string for
the Image.

You can apply any transformation in the scope of Vectorgebra
to the Image's tensor.

You can save images by the "imsave()" function. This function
again does not belong to the Image class. You need to provide
a proper path and an Image object to this function.

All optional PNG headers are ignored for now. Some images may
look different. PLTE chunk is also ignored. Related data is read
then ignored. Required supports will come in the future. 
