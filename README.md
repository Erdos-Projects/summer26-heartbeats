# Problem Definition

Devices like smartwatches are often used to track physical activity, helping users meet their exercise goals. This tracking relies on accurate inference of the physical activity being performed from data available to the device. These data may look similar between activities which the user would prefer to distinguish, making quick and accurate activity classification challenging.

**In this project, we build models capable of classifying physical activities based on data accessible by consumer smart devices.** Once trained, this model can run live on consumer devices such as smart watches, classifying periods of activity and displaying progress towards the user's movement goals. In addition to consumer use, data classified by this model could have clinical applications to long-term medical tests such as holter monitors. Clinical datasets are often analyzed after collection is complete, making live classification beneficial but not essential. Key stakeholders and their priorities are:

* Executive staff at technology companies such as Apple, Samsung, and Google
    * The model should be *accurate*
    * The model should be computationally *inexpensive*
    * The model should classify activities *quickly*
* Medical professionals 
    * The model should be *accurate*
    * The model should be computationally *inexpensive*

# Data Gathering

We train and validate our model with the publicly available [PAMAP2 Physical Activity Monitoring](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) dataset. The compressed dataset is approximately 650 MB in size, and was be acquired with a one-time download.

This dataset was collected for 9 subjects while they performed a variety of physical activities. The data includes heart rate and body temperature, alongside measurements of the 3-D acceleration, 3-D magnetic field, and 3-D rotation speed recorded by gyroscope. All measurements except heart rate were made with inertial measurement units (IMU) placed on the subjects' wrist, ankle, and chest. Over 10 hours of data were recorded for each subject, and were classified into 18 physical activity categories by time interval.

# Data Assessment

With over 10 hours of data for each subject, we expect to have sufficient data for random forest classification. IMU measurements are sampled at 100 Hz, and heart rate is sampled at 9 Hz. This means each subject has at least $3.6\times10^{6}$ measurements from each of the three IMUs, and at least $3.24\times10^5$ heart rate measurements.

The number of subjects, however, may limit our analysis. The sample of only 9 subjects (all of whom are in the age range of 23-31 years, and only one of whom is female) is likely insufficient to model the variation in physical activity metrics among the population of smartwatch users. This may mean we should focus on building a model which is trained on a set of each individual's data, rather than a general model which applies to all users. This would require the user to calibrate their personal device by manually classifying activities on its first use.

# KPI Definition

The key performance indicators for this project, in order of importance, are:

* Accuracy of classification
    * Does the model accurately classify physical activities?(Confusion matrix, Accuracy score, per person accuracy)
* Classification speed
    * How many samples are required to classify an activity?
* Computational efficiency
    * How much of a smartwatch's computational resources will this model require?
    * What are the mimimum data features needed to tell the activity apart and still keep its accuracy?

# Methods

Our preprocessing pipeline is fully contained in ML_pipeline/data_preprocessing. This pipeline locates segments of continuous activity, interpolates missing values where appropriate, and applies a Butterworth filter to exclude data outside the typical frequency range of human activity. The pipeline then extracts hand-crafted features from the cleaned segments, and saves them to a features.csv file (not included in this repository due to its large size).

We test three classical machine learning models (random forest, support vector machines, and K nearest neighbor) and one deep learning model (a convolutional neural network) against a simple baseline model. Training scripts and results for all five of these models can be found in model_training_code.

To run our pipeline, the user must first download the [PAMAP2 Physical Activity Monitoring](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring). Then the user can generate the features.csv file, and replicate our model training process.

# Results

We find that traditional machine learning models all classify activities with high accuracy (~90%), significantly better than our baseline model (53%). Our Convolutional Neural Network also clears the baseline, but lags behind the traditional models (83%).

# Future Work

This project focused on our first KPI, model accuracy. With our training pipeline we can perform further tests address our additional KPIs, such as benchmarking the computational efficiency of our models. We can also test the amount of training data from an individual necessary for accurate classification of their activities, which is important for the calibration period in newly bought commercial devices. Some of these tests may require larger datasets that are more representative of the general population.

An additional objective would be to restrict our data to more accurately match the information collected by commercial smart devices. To do this, we would exclude the chest and ankle IMUs, and reduce the sampling frequency of our data to match consumer devices. If we vary the IMU being used (such as keeping either the chest or ankle device instead of the wrist), we could also determine if there is an optimal device placement for activity classification. This could be of interest for consumer devices dedicated to exercise tracking, and for medical devices which are often not worn on the wrist.