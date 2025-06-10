*********************************************************************************************************************************************
* For non-beamforming methods, run train_compare_spectral.py. All other python files perform various preprocessing tasks or are deprecated. *
*********************************************************************************************************************************************

Final Training Loop and Dependencies:

* cnn.py contains all neural network definitions (not just CNNs)
* preprocessing.py contains helper functions for preprocessing
* loaddata.py loads raw data.
* load_dataset.py calls loaddata.py and preprocessing.py, splits into windows, performs FFT, and saves.
* tv_split.py splits the dataset into training and validation datasets.
* SproulToVLA.txt contains range definitions in a tabular format. Please keep for running.
* train_compare_spectral.py contains all FINAL training loops. RUN THIS FILE. /!\

Other Files (from earlier iterations of coding):
* train_compare.py contains training code for only TD-CNN with bandpassed input.
* fft_fig.py plots FFT magnitude for a window.
* spect.py and spectro.py contains spectrogram code for raw data (each is independent of the other file)
* plotting.py plots a single window of data for visualization purposes.



*********************************************************************************
For beamforming methods, all code is contained in BF.ipynb. RUN THIS FILE /!\
*********************************************************************************
