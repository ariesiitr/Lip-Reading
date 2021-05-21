## LIP-Reading-IITR

### Abstract:

Human lip-reading is a challenging task. It requires not only knowledge of underlying language but also visual clues to predict spoken words. Experts need certain level of experience and understanding of visual expressions learning to decode spoken words. Now-a-days, with the help of deep learning it is possible to translate lip sequences into meaningful words.

### Background:

Lipreading helps people understand more speech by watching for and identifying mouth movements that are associated with speech. Being able to see speech helps people communicate better, especially in challenging listening environments like when there is background noise.

### Datasets Used:

* MIRACL dataset was used small model preparation wherein words and phrases were simply classified based on the data (frames for word which was taken from Kaggle ([link](https://www.kaggle.com/apoorvwatsky/miraclvc1))

	* It’s a new LR dataset consisting on 1500 word (15 persons×10 words×10 instances) and 1500 phrases (15 persons×10 phrases×10 instances). The dataset covers words like navigation, connection, etc., and everyday phrases like Nice to meet you, I love this game, etc. The Kinect sensor was used to acquire 2D images and depth maps with a resolution of 640 × 480 pixels and at an acquisition rate of 15 fps. The distance between the speaker and the Kinect is about 1m. 

* GRID dataset was used for training the Lip-Net model which along with audio alignment and connectionist temporal classification (CTC) loss provides state of the art results. ([link to the dataset](http://spandh.dcs.shef.ac.uk/gridcorpus/)).

	* GRID is a large multitalker audio-visual sentence corpus to support joint computational-behavioral studies in speech perception. In brief, the corpus consists of high-quality audio and video (facial) recordings of 1000 sentences spoken by each of 34 talkers (18 male, 16 female). Sentences are of the form "put red at G9 now".  The corpus, together with transcriptions, is freely available for research use. 

### Technical Requirements and Setup:
* MIRACL – We simply used Kaggle’s inbuilt notebook and GPU for preprocessing and training purpose which could be found in MIRACL_FILES above.

* GRID – Major computation was done using AWS Sage-maker with S3 bucket ([link to our bucket with data](https://s3.console.aws.amazon.com/s3/home?region=ap-south-1)) as the data storage unit for the GRID dataset.
	* Instance Used - ml.m4.xlarge with 4 CPU, 16GB RAM and 1000GB space.
	* Pricing of Sagemaker can be found at this [link](https://aws.amazon.com/sagemaker/pricing/). For detailed information on how to start using AWS and Sagemaker follow these links. ([link1](https://adamtheautomator.com/upload-file-to-s3/), [link2](https://www.pluralsight.com/guides/build-your-first-deep-learning-solution-with-aws-sagemaker))
	* To setup AWS instance, first upload the data into an S3 bucket, then use Setup.ipynb for setting up and unzipping the data for further use.
	The directory will look like the below picture with additional files about which more description is given in the lower section.

![assets](https://github.com/parthchhabra0611/LIP-Reading-IITR/blob/main/directory.jpg)	
	 


### Preprocessing the data:

The [Dlib](https://pypi.org/project/dlib/) library was used to extract the major keypoints of mouth at each timestep. Based on the keypoints (using extreme left, right, top and bottom coordinates), we cropped each frame and captured the mouth and lips. 

* MIRACL – For this dataset the preprocessed frames were saved in .jpg format using Pillow framework in the working directory.

* GRID - These pre-processed frames were then saved as .npy files in the x_data/ folder. 
This data requires audio preprocessing. For alignment of audio files to the video files, the connectionist temporal classification (CTC) loss was used. It is widely used in modern speech recognition as it eliminates the need for training data that aligns inputs to target outputs. Given a model that outputs a sequence of discrete distributions over the token classes (vocabulary) augmented with a special “blank” token, CTC computes the probability of a sequence by marginalising over all sequences that are defined as equivalent to this sequence. This simultaneously removes the need for alignments and addresses variable-length sequences.

	* More information can be found in this [research paper](https://arxiv.org/pdf/1611.01599.pdf) which considers use of Lipnet and importance of CTC for obtaining best results with GRID dataset.

	* So, for incorporating CTC, audio_to_ctc_labels file was used. This made use of pretrained Wave2Vec transformer to convert audio files to ctc texts which gave characters from a-z for each spoken character and an underscore ( _ ) for the timesteps when the speaker remains quiet. All the audio files were processed for each speaker and stored as a csv file namely ‘y_labels.csv’. 
	For the implementation of Wave2Vec, you may refer [this article](https://www.kdnuggets.com/2021/03/speech-text-wav2vec.html).

### Modelling and Training:

* MIRACL – The VGGFace model was used which is trained and evaluated on benchmark face recognition datasets, demonstrating that the model is effective at generating generalized features from faces. The VGGFace model is described by Omkar Parkhi in the 2015 paper titled [“Deep Face Recognition”](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf).
	* The model file is present above by the name ‘VGGFace_training_validation.ipynb’ in MIRACL_FILES. A time distributed layer, an LSTM layer and final dense layer was used on the features produced by VGGFace and classification was done.

* GRID – Lipnet model was used on the processed videos and corresponding ctc labels. LipNet is a neural network architecture for lipreading that maps variable-length sequences of video frames to text sequences, and is trained end-to-end. For more information refer this [research paper](https://arxiv.org/pdf/1611.01599.pdf). 
	* The model file is present above by the name ‘lipnet.ipynb’. The training file is present as ‘training.ipynb’ and this file imports Lipnet model and all the preprocessed data and trains the model. The outputs i.e. the predictions given by the model on the validation data can be decoded using a decoder which converts ctc labels back to text.

### Result:

* MIRACL - Metric used for evaluation was accuracy.
	* Training accuracy : 97%
	* Validation accuracy : 46.15%
	* Test accuracy : 25%
* GRID  - Metric for evaluation was Word Error Rate (WER)
	* ![assets2](https://github.com/parthchhabra0611/LIP-Reading-IITR/blob/main/wer_img.jpg)	
    * Training WER : 22.76%
    * Validation WER : 38.21%
    * Test WER : 50.42%

### Team Members:
* [Parth Chhabra](https://github.com/parthchhabra0611)
* [Kushagra Babbar](https://github.com/kush1920)
* Dhaval Kanani
* Chirag Sethiya

### Future Improvements:

* Used data augmentation with jitter, random flipping of videos.
* Create custom audio to ctc model inplace of pretrained transformer.
* Train for more epochs.
* Train on LSR dataset for longer sentences.

