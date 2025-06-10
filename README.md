*********************************************************************************************************************************************
* For non-beamforming methods, run load_dataset.py and then train_compare_spectral.py. All other python files perform various preprocessing tasks or are deprecated. *
*********************************************************************************************************************************************

/!\ The S5 EXPERIMENT DATA (non-noisy) is too large for GitHub. Please download it from this link:
https://drive.google.com/file/d/1bCmopIiqDT2-MPvg0TQczaCgK-yaSg7m/view?usp=sharing
Leave data file s5.mat in the project directory.

The S59 EXPERIMENT DATA (noisy) is smaller in size so it is in the GitHub Repo. It is labeled "s59.mat".

Final Training Loop and Dependencies:

* cnn.py contains all neural network definitions (not just CNNs)
* preprocessing.py contains helper functions for preprocessing
* loaddata.py loads raw data.
* load_dataset.py calls loaddata.py and preprocessing.py, splits into windows, performs FFT, and saves. /!\ RUN THIS FILE before train_compare_spectral.py. If you cannot find any ".npz" files, you must run this file first.
* tv_split.py splits the dataset into training and validation datasets.
* SproulToVLA.txt contains range definitions in a tabular format. Please keep for running.
* train_compare_spectral.py contains all FINAL training loops. RUN THIS FILE. /!\

Other Files (from earlier iterations of coding):
* train_compare.py contains training code for only TD-CNN with bandpassed input.
* fft_fig.py plots FFT magnitude for a window.
* spect.py and spectro.py contains spectrogram code for raw data (each is independent of the other file)
* plotting.py plots a single window of data for visualization purposes.



*********************************************************************************
For beamforming methods, all code is contained in BF_v2.ipynb. RUN THIS FILE /!\ An animation is in Beamforming UKF.mp4.
*********************************************************************************
