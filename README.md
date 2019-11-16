Speaker Recognition One Shot Learning
---------------------------------------------------------------

#### Speaker Recognition Overview
Speaker recognition is person identification given an audio file. Used to answer the question "Who is speaking?" \
Speaker verification (also called speaker authentication) is similar to speaker recognition, but instead of identifying the speaker it gives an indication whether the speaker is who we claimed.\
Speaker Verification is considered to be a little easier (binary classification task) than speaker recognition (multi class classification).

#### One Shot Learning Overview
One-shot learning is an object categorization problem in computer vision. Whereas most machine learning based object categorization algorithms require training on hundreds or thousands of images and very large datasets, one-shot learning aims to learn information about object categories from one, or only a few, training images.

---------------------------------------------------------------
#### How to Use?
* clone the repo
* cd to the cloned dir
* `python main.py`

#### Optional Args:
* `--model-dir-path` - path to saved model directory
* `--output-dir` - path to output folder (contains figures results etc.)
* `--weight-name` - model's weight file name
* `--model-name` - model's file name
* `mode` - train / validation mode
* For more optional arguments use `--help`