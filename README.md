Recommendation system based on personal preferences in appearance
# Abstract
We made a recommendation system based on preferences in the appearance of individual users. We use content base filtering instead of collaborative filtering.
# Model
First, we extract facial features using encoders, representing the face in a vector space in which arithmetic operations are applicable. Such transformations allow to reveal high abstraction features such as hair color,smile, or even gender.
To begin with, we give the user a set of random photos for evaluation. When there is enough data for training, our model is used for further recommendations and can be trained online after this point. We implemented the Siamese network to solve the problem of face recognition and finding vector representations of our photos.
$ Similarity(v1,v2) = || v1  - v2 || $
First of all, we found all the V views for all photos using a neural network and built a matrix R of the distances between all the available photos.
$ R(i,j) = Similarity(v_i,v_j) $
Now we create the preferences vector L. We count the sum of the deviations from the vector presentations of the liked photos
$ L(i)=\sum_{i=1}^{n} R(i,k) $
Thus, we get how close this photo is to the user’s preferences in appearance. It remains for us to sort the vector L and use the minimum values.
If a person has several types of preferences, then we can use clustering using the Gaussian Mixture Model.
# Dependencies
numpy
pandas
[face_recognition](https://github.com/ageitgey/face_recognition)
flask
# Dataset
[Selfie Data Set](https://www.crcv.ucf.edu/data/Selfie/) contains 46,836 selfie annotated images
# How to use
> python server.py