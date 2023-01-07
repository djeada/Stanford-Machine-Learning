## Application Example - Photo OCR.
Optical character recognition, or OCR, is the process of teaching computers to interpret text in images. A machine learning pipeline is a series of steps that helps to solve a problem, such as detecting pedestrians or text in an image. One way to do this is to create a training set of positive and negative examples, train a classifier using this data, and then apply the classifier to an image using a sliding window method. The final stage of the process, character recognition, involves using a training set of character images to teach the system to recognize characters based on image patches. This training data can be enhanced through techniques such as data synthesis, which involves adding distortion to existing data. The goal of these pipelines is to allow computers to better understand and interpret digital images.

## Problem description and pipeline

* Consider how a complex system may be built.
* The idea of a machine learning pipeline.
* Machine learning applied to real-world issues and artificial data synthesis.

What is the photo OCR problem?

* Photo OCR = photo optical character recognition.
* Getting computers to interpret digital images is one notion that has piqued the curiosity of many people.
* The idea behind picture OCR is to teach computers to interpret text in images.

Pipeline - a sequence of separate modules, each of which may be a machine learning or data processing component.

![ocr_pipeline](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/ocr_pipeline.png)

## Pedestrian detection


* We're looking for pedestrians in the picture.
* A common aspect ratio for a standing human is 82 x 36.
* Obtain a training set of both positive and negative examples (1000 - 10 000).


![pedestrians_training_set](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/pedestrians_training_set.png)

* Now that we have a new image, how do we identify pedestrians within it?
* Start by taking a rectangular 82 x 36 patch in the image.
* Run patch through classifier (returns 1 or 0).
* After that, move the rectangle slightly to the right and re-run the program.
* Repeat until you've covered the whole image.
* Hopefully, by varying the patch size and rastering over the image frequently, you will finally recognize all of the pedestrians in the image.


![pedestrians](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/pedestrians.png)

## Text detection example


* We generate a labeled training set with positive examples (some type of text) and negative examples (not text), similar to pedestrian detection.
* After training the classifier, we apply it to an image using a sliding window classifier with a set rectangle size.
* Obtain a training set of both positive and negative examples (1000 - 10 000).
* The white region indicates where the text recognition algorithm believes text exists, while the varying shades of gray correlate to the likelihood associated with how certain the classifier is that the section includes text

![text_detection](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/text_detection.png)

* Take the classifier output and apply an expansion algorithm that extends each of the white areas.
* Examine the linked white patches in the picture above. Draw rectangles around those that make sense as text (tall narrow boxes don't).
* This example misses a piece of text on the door because the aspect ratio is wrong.

![expansion_algorithm](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/expansion_algorithm.png)

### Stage two is character segmentation


* To navigate along text areas, use a 1-dimensional sliding window.
* Does each window snapshot resemble the separation of two characters?
* Insert a split if yes.
* If not, proceed.

![character_segmentation](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/character_segmentation.png)

## Character recognition as an example of data synthesis


* The goal is for the system to recognize a character based on an image patch.
* Consider the photos to be grayscale (makes it a bit easer).
* Where can I find training data?
* Modern computers frequently include a large font library, and there are several free font libraries available on the internet.
* Take characters from different fonts and place them on random backgrounds to get more training data.
* Another approach is to add distortion into existing data, such as warping a character.

![characters](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/characters.png)
