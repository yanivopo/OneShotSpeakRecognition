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

The original data can be found in `http://www.robots.ox.ac.uk/~vgg/data/voxceleb` (~30 GB).
#### pre process of woxceleb 

##### split the data to 3 sec and stft transfrom: 
`python data_process_util.py -i path/to/woxceleb/dataset -o path/to/outputdir ` 
##### create offline triplet:  
`python oneShot/data_generator.py -i path/to/preProcess/data `
##### Optional Args: 
* `--batch-size` - # batch size.
* `--number-of-batch-train` - # different batch for the train.
* `--number-of-batch-valid` - # different batch for the validation.

### To Run example: 
* `python main.py`

#### Optional Args:
* `--train-data-path` - path to pre-process data the same dir for data_process_util.py output.
* `--output-dir` - path to output folder. (contains figures results etc.)
* `--weight-name` - model's weight file name.
* `--model-name` - model's file name.
* `mode` - train from scratch / load existing model.
* For more optional arguments use `--help`.