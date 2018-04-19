# dl_demos
Collection of deep learning demos 

### Repository content ###
  
The repository contains demos of different frameworks and different use cases, on public datasets.
  
* The folder waveform_clustering contains approaches for clustering of time series:  

    1. The subfolder Human_Activity_Recognition contains kmeans applied on mobile phone sensor signals to cluster them by the activity of the person (walking, sitting etc.) The original data and supervised solution can be found here: https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition    

    2. The subfolder ECG contains LSTM autoencoder approach applied on ECG signals, to cluster them by heart condition. The ground truth labels are known therefore the accuracy can be verified. The original example can be found here: https://github.com/RobRomijnders/AE_ts  

    3. The subfolder MNIST contains clustering on the MNIST dataset, by Variational autoencoders, with or without Convolutional layers. These aproaches were used for the final solution. The examples are to be found here:  
        * without convolutional layers: https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py  
        * with convolutional layers: https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder_deconv.py  
    
  
* signal_classification folder contains:  
    * The HAR folder contains demos of different frameworks classifying the Human activity recognition signals (supervised case). The original example can be found here: https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition  

* time_series folder contains a CNTK demo of a time series prediction trained on IoT sensor data. Source: https://github.com/Microsoft/CNTK/blob/v2.4/Tutorials/CNTK_106B_LSTM_Timeseries_with_IOT_Data.ipynb  
  
* classification_iris contains demos of different frameworks classifying the Iris dataset (https://archive.ics.uci.edu/ml/datasets/iris).  
  
* classification_mnist contains demos of different frameworks classifying the hand written digits (yann.lecun.com/exdb/mnist/), by using CNN.  
  
* GAN contains a demo of a generative adversarial network for distribution fitting/generation  
  
* data contains the data for all of the above examples
  
### Repository owner ###
monica.rafaila@gmail.com
