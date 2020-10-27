import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
import copy
import os
import random
import sys

from sklearn import preprocessing
from scipy.stats import pearsonr

from keras import backend as K
from keras.models import Sequential, load_model, save_model, Model
from keras.layers import Input, Dense, Masking, LSTM, TimeDistributed, Dropout
from keras.layers.noise import GaussianNoise
from keras.optimizers import RMSprop
from keras.callbacks import History

import warnings

from numpy.random import seed
from tensorflow import set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument("--DB", type=str) # database code (SEW, CIT, UUD, IEC, SEW)
parser.add_argument("--SN", type=str) # scenario (BAS, SWT, FLF)
parser.add_argument("--SO", type=int) # sparsing original speaker
parser.add_argument("--SI", type=int) # sparsing interlocutor
parser.add_argument("--ED", type=str) # emotional dimension, arousal/valence
parser.add_argument("--RS", type=int, default=1) # random seed for partitioning

args = parser.parse_args()

DB = args.DB
SN = args.SN
sparsing_interlocutor = args.SI
sparsing = args.SO
ED = args.ED
RS = args.RS

seed(RS)
set_random_seed(RS)

warnings.simplefilter(action='ignore', category=FutureWarning)

results_folder = DB + '_' + ED + '_' + str(RS)

# Check if results folder exists. If not - create one
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

def add_timesteps (X, Y, R, index, timewindowX, timewindowY, sc, labels_forward_shift):
    # This function transforms 2D arrays of features and 1D arrays of labels into 3D arrays in time-continuous window length fashion

    # Inputs:
    # X - 2D features numpy array with shape [samples, features]
    # Y - 1D labels numpy array with shape [samples]. It contains values to be predicted, e.g. valence or arousal.
    # R - 1D recordings array with shape [samples]. It contains names of recodrings and needed for correct window extraction when data from several recordings is concatenated
    # index - 1D index array with shape [samples]. It contains unique index of each time step.
    # timewindowX - integer, number of time steps to be considered for features
    # timewindowY - integer, number of time steps to be considered for labels
    # sc - integer, sparsing coefficient, see here for more detail: http://www.lrec-conf.org/proceedings/lrec2018/pdf/923.pdf
    # labels_forward_shift - float between 0 and 1, 0 means that features and labels will be taken for the same frames
    
    # Note that features and labels should be of the same sample length and this should have been controlled on previous steps of preprocessing

    # Outputs:
    # X_cont - 3D features numpy array with shape [samples, timesteps, features]. Ready-to-use in further modeling
    # Y_cont - 3D labels numpy array with shape [samples, timesteps, 1]. Ready-to-use in further modeling
    # Y_cont_index - 2D index numpy array with shape [samples, timesteps]. Needed for smooting procedure later
    
    c = 1 - labels_forward_shift
    
    # Create arrays for time-continuous data     
    X_cont = np.zeros((X.shape[0], timewindowX, X.shape[1]))
    Y_cont = np.zeros((Y.shape[0], timewindowY))
    Y_cont_index = np.array([([None] * timewindowY)] * index.shape[0])
    
    # Loop over samples 
    for i in range (X.shape[0]):
        # Loop over time steps
        for j in range (0, timewindowX):
            # Check if we are not exceeding limits of indices for array
            if (i - sc*timewindowX + sc*j + sc) >= 0:
                # Check if recordings are the same (not to take frames from different recordings into one sample)
                if R[i] == R[i - sc*timewindowX + sc*j + sc]: 
                    # Assign timesteps
                    X_cont[i,j,:] = X[i - sc*timewindowX + sc*j + sc,:]
                    Y_cont_index[i,j] = index[i - sc*int(np.floor(timewindowY*c)) + sc*j + sc*int(0 ** (1 - c))]
        
        # Do the same for labels, but up to timewindowY (if they differ)            
        for j in range (0, timewindowY):
            if 0 <= (i - sc*np.floor(timewindowY*c) + sc*j + sc*(0 ** (1 - c))) < Y.shape[0]: 
                if R[i] == R[i - sc*int(np.floor(timewindowY*c)) + sc*j + sc*int(0 ** (1 - c))]:
                    Y_cont[i,j] = Y[i - sc*int(np.floor(timewindowY*c)) + sc*j + sc*int(0 ** (1 - c))]
                    Y_cont_index[i,j] = index[i - sc*int(np.floor(timewindowY*c)) + sc*j + sc*int(0 ** (1 - c))]
                  
    return X_cont, Y_cont, Y_cont_index

def smoother (prediction, index, index_cont, smoothing_windowsize, sc, labels_forward_shift):
    # This funstion smoothes predictions for time steps obtained for different samples
    
    # Inputs:
    # prediction - 2D prediction numpy array with shape [samples, timestamps]
    # index - 1D index array with shape [samples]. It contains unique index of each time step.
    # index_cont - 2D time-continuous index array with shape [samples, timestamps]. Corresponds to "prediction" array
    # smoothing_windowsize - integer, number of time steps to smooth over
    # sc - integer, sparsing coefficient, see here for more detail: http://www.lrec-conf.org/proceedings/lrec2018/pdf/923.pdf
    # labels_forward_shift - float between 0 and 1, 0 means that features and labels will be taken for the same frames
  
    # Outputs:
    # prediction_smooth - 1D numpy array of smoothed predictions with shape [samples]

    # Create array for smooth predictions
    prediction_smooth = np.zeros((len(index)))
    
    # Loop over samples
    for i in range(0,len(index)):
        counter = 0
	
	# Restrict search space by setting lower and upper index boundaries
        lower_bound = max(0,i - smoothing_windowsize*sc * max(1,labels_forward_shift))
        upper_bound = i + smoothing_windowsize*sc * max(1,labels_forward_shift)
	
	# Find predictions for current index
        locate = np.where(index_cont[lower_bound:upper_bound] == index[i])
	
	# Sum up over them
        for loc_x in range(0, locate[0].shape[0]):
            for loc_y in range(0, locate[1].shape[0]):
                prediction_smooth[i] += prediction[lower_bound + locate[0][loc_x],locate[1][loc_y]]
                counter += 1
		
	# Calculate mean
        prediction_smooth[i] /= counter 
        
    return prediction_smooth

def CCC_loss_tf(y_true, y_pred):
    # Concordance correlation coefficient (CCC)-based loss function - using non-inductive statistics
    # Used in TensorFlow based operations

    y_true_mean = tf.reduce_mean(y_true)
    y_pred_mean = tf.reduce_mean(y_pred)
    
    y_true_var = tf.reduce_mean(tf.square(tf.subtract(y_true, y_true_mean)))
    y_pred_var = tf.reduce_mean(tf.square(tf.subtract(y_pred, y_pred_mean)))
    
    cov = tf.reduce_mean((y_true - y_true_mean) * (y_pred - y_pred_mean))
    
    CCC = 2 * cov / (y_true_var + y_pred_var + tf.square(y_true_mean - y_pred_mean))
    
    loss = 1 - CCC

    return loss

def CCC_metric(ser1, ser2):
    # Concordance correlation coefficient (CCC)-based metric function
    if ser1.shape != ser2.shape :
        print("Series have different lengths")
    else:
        CC = pearsonr(ser1, ser2)
        CCC = 2*CC[0]*np.std(ser1)*np.std(ser2)/(np.var(ser1) + np.var(ser2) + (np.mean(ser1, axis=0) - np.mean(ser2, axis=0)) ** 2)
    return CCC

def create_model(X_c, timewindow):
    # This function creates a RNN-LSTM model to be trained further
    # TODO: rewrite using dictionary
    model = Sequential()
    model.add(GaussianNoise(0.1, input_shape=(timewindow, X_c.shape[2])))
    model.add(LSTM(80, return_sequences=True, activation='tanh', implementation=0))
    model.add(Dropout(0.3))
    model.add(LSTM(60, return_sequences=False, activation='tanh', implementation=0))
    model.add(Dropout(0.3))
    model.add(Dense(timewindow, activation='linear'))

    RMS = RMSprop(lr=0.005, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss=CCC_loss_tf, optimizer=RMS)  # CCC-based loss function

    return model


def predict (trained_model, X_c, I_c, base_index):
    # This function uses trained model to perform predict either for train, development or test subset

    # Inputs:
    # trained_model - trained Keras RNN-LSTM model
    # X_c - 3D features numpy array with shape [samples, timesteps, features]
    # I_c - 2D index numpy array with shape [samples, timesteps]
    # base_index - 1D index array with shape [samples]
  
    # Outputs:
    # P_train_clean - 1D numpy array of smoothed and cleaned predictions with shape [samples]

    # Obtain continuous predictions
    P_train_c = trained_model.predict(X_c, verbose=0)

    # Smooth predictions to 1D array
    P_train = smoother(P_train_c, base_index, I_c, timewindow, sparsing, 0)
    
    # Check for NaNs
    P_train_clean = np.nan_to_num(P_train)
    
    return P_train_clean

def load_data(db_name):
    # This function loads data from *.csv files of features and labels

    # Inputs:
    # db_name - three-letter index of the database. E.g. "SEW" = SEWA database.

    # Outputs:
    # features - 2D pandas DataFrame of features with shape [samples, features]
    # labels - 1D pandas DataFrame of labels with shape [samples]
    # recordings - 1D pandas DataFrame of recording indices with shape [samples]

    print('Load data')

    # SEWA database
    if db_name == 'SEW':
        path = 'data/'
	# Load features
        features = pd.read_csv(path + 'SEW_features_egemaps_0.3_10Hz.csv', index_col=0)
	
	# Load labels
        if ED == 'arousal':
            labels = pd.read_csv(path + 'SEW_labels_10Hz.csv', index_col=0).arousal.to_frame()
        elif ED == 'valence':
            labels = pd.read_csv(path + 'SEW_labels_10Hz.csv', index_col=0).valence.to_frame() 
	
	# Load recordings
        recordings = pd.DataFrame([x[:7] for x in labels.index], index=labels.index)
	
    # UUDB database
    elif db_name == 'UUD':
        path = 'data/'
	# Load features
        features = pd.read_csv(path + 'UUD_features_LLD_10Hz.csv', index_col=0)
	
	# Load labels
        if ED == 'arousal':
            labels = pd.read_csv(path + 'UUD_labels_TC.csv', index_col=0).arousal.to_frame()
        elif ED == 'valence':
            labels = pd.read_csv(path + 'UUD_labels_TC.csv', index_col=0).valence.to_frame() 
	
	# Load recordings
        recordings = pd.DataFrame([x[:7] for x in labels.index], index=labels.index)
	
    # CreativeIT database
    elif db_name == 'CIT':
        path = 'data/'
	# Load features
        features = pd.read_csv(path + 'CIT_features_LLD_10Hz.csv', index_col=0)
	
	# Load labels
        if ED == 'arousal':
            labels = pd.read_csv(path + 'CIT_labels_10Hz.csv', index_col=0).arousal.to_frame()
        elif ED == 'valence':
            labels = pd.read_csv(path + 'CIT_labels_10Hz.csv', index_col=0).valence.to_frame()
	
	# Load recordings
        recordings = pd.DataFrame([x[:7] for x in features.index], index=features.index)

    # CreativeIT database
    elif db_name == 'SEM':
        path = 'data/'
	# Load features
        features = pd.read_csv(path + 'SEM_features_LLD_10Hz_d.csv', index_col=0)
	
	# Load labels
        if ED == 'arousal':
            labels = pd.read_csv(path + 'SEM_labels_d.csv', index_col=0).arousal.to_frame()
        elif ED == 'valence':
            labels = pd.read_csv(path + 'SEM_labels_d.csv', index_col=0).valence.to_frame()
	
	# Load recordings
        recordings = pd.DataFrame([x[:7] for x in labels.index], index=labels.index)

    # IEMOCAP database
    elif db_name == 'IEC':
        path = 'data/'
	# Load features
        features = pd.read_csv(path + 'IEC_features_LLD_10Hz.csv', index_col=0)
	
	# Load labels
        if ED == 'arousal':
            labels = pd.read_csv(path + 'IEC_labels_TC.csv', index_col=0).arousal.to_frame()
        elif ED == 'valence':
            labels = pd.read_csv(path + 'IEC_labels_TC.csv', index_col=0).valence.to_frame() 
	
	# Load recordings
        recordings = pd.DataFrame([x[:7] for x in labels.index], index=labels.index)

    # Check intersection of features and labels
    ind = features.index.intersection(labels.index)

    # Subset each DataFrame accordingly
    features = features.loc[ind]
    labels = labels.loc[ind]
    recordings = recordings.loc[ind]
        
    # Check is the length is the same. This may still be an issue even after subsetting with "ind" list, as there might be duplicates in data
    if features.shape[0] == labels.shape[0] and features.shape[0] == recordings.shape[0]:
        print('Shapes are OK!')
    
    # If sizes do not match, print them 
    else:
        print(features.shape[0])
        print(labels.shape[0])
        print(recordings.shape[0])
            
    return features, labels, recordings

def get_index_by_code(recordings, partition):
    # This function selects recording indices corresponding to particular subset (train/dev/test)

    # Inputs:
    # recordings - pandas DataFrame with indices and recorging names of the data
    # partition - list of recordings corresponding to particular subset

    # Outputs:
    # index - 1D array of indices corresponding to particular subset

    for t in range(0,len(partition)):
        temp = recordings[recordings == partition[t]].dropna().index

        if t == 0:
            index = temp
        else:
            index = index.append(temp)
            
    return index

def preprocess_features(scenario, features, labels, recordings, train_index, dev_index, test_index, DB, sparsing_interlocutor=6):
    if scenario == 'BAS':
        features_train = features.loc[train_index]
        features_dev = features.loc[dev_index]
        features_test = features.loc[test_index]

        scaler = preprocessing.StandardScaler()

        features_norm = scaler.fit_transform(features_train.as_matrix())

        features_norm = pd.DataFrame(features_norm, index=features_train.index)

        features_norm_dev = pd.DataFrame(scaler.transform(features_dev.as_matrix()), index=features_dev.index)
        features_norm_test = pd.DataFrame(scaler.transform(features_test.as_matrix()), index=features_test.index)

        labels_train = labels.loc[train_index]
        labels_dev = labels.loc[dev_index]
        labels_test = labels.loc[test_index] 

        #print('Subsampled')

        #print('Contextual preprocessing')
        [X_c, Y_c, I_c] = add_timesteps(features_norm.as_matrix(), labels.loc[train_index].as_matrix(), recordings.loc[train_index].as_matrix(),
                                        features_norm.index, timewindow, timewindow, sparsing, 0)

        [X_c_dev, Y_c_dev, I_c_dev] = add_timesteps(features_norm_dev.as_matrix(), labels.loc[dev_index].as_matrix(), recordings.loc[dev_index].as_matrix(),
                                        features_norm_dev.index, timewindow, timewindow, sparsing, 0)

        [X_c_test, Y_c_test, I_c_test] = add_timesteps(features_norm_test.as_matrix(), labels.loc[test_index].as_matrix(), recordings.loc[test_index].as_matrix(),
                                        features_norm_test.index, timewindow, timewindow, sparsing, 0)

        """Y_c = np.reshape(Y_c, (Y_c.shape[0], Y_c.shape[1], 1))
        Y_c_dev = np.reshape(Y_c_dev, (Y_c_dev.shape[0], Y_c_dev.shape[1], 1))
        Y_c_test = np.reshape(Y_c_test, (Y_c_test.shape[0], Y_c_test.shape[1], 1))"""
    
    ########################################## SWITCHING LABELS ############################################################
    elif scenario == 'SWT':
        print('Switching labels')

        # SWITCHING LABELS
        original_index = copy.deepcopy(labels.index.values)
        swiched_index = copy.deepcopy(labels.index.values)
        for i in range (swiched_index.shape[0]):
            splitted = list(original_index[i])

            if splitted[3] == '1':
                splitted[3] = '2'
            elif splitted[3] == '2':
                splitted[3] = '1'

            swiched_index[i] = ''.join(splitted)

        # switching for all
        labels.index = swiched_index

        ########### END OF SWITCHING ################

        features_train = features.loc[train_index]
        features_dev = features.loc[dev_index]
        features_test = features.loc[test_index]

        scaler = preprocessing.StandardScaler()

        features_norm = scaler.fit_transform(features_train.as_matrix())

        features_norm = pd.DataFrame(features_norm, index=features_train.index)

        features_norm_dev = pd.DataFrame(scaler.transform(features_dev.as_matrix()), index=features_dev.index)
        features_norm_test = pd.DataFrame(scaler.transform(features_test.as_matrix()), index=features_test.index)

        labels_train = labels.loc[train_index]
        labels_dev = labels.loc[dev_index] 
        labels_test = labels.loc[test_index] 

        print('Subsampled')

        print('Contextual preprocessing')
        [X_c, Y_c, I_c] = add_timesteps(features_norm.as_matrix(), labels.loc[train_index].as_matrix(), recordings.loc[train_index].as_matrix(),
                                        features_norm.index, timewindow, timewindow, sparsing, 0)

        [X_c_dev, Y_c_dev, I_c_dev] = add_timesteps(features_norm_dev.as_matrix(), labels.loc[dev_index].as_matrix(), recordings.loc[dev_index].as_matrix(),
                                        features_norm_dev.index, timewindow, timewindow, sparsing, 0)
        
        [X_c_test, Y_c_test, I_c_test] = add_timesteps(features_norm_test.as_matrix(), labels.loc[test_index].as_matrix(), recordings.loc[test_index].as_matrix(),
                                        features_norm_test.index, timewindow, timewindow, sparsing, 0)


        """Y_c = np.reshape(Y_c, (Y_c.shape[0], Y_c.shape[1], 1))
        Y_c_dev = np.reshape(Y_c_dev, (Y_c_dev.shape[0], Y_c_dev.shape[1], 1))
        Y_c_test = np.reshape(Y_c_test, (Y_c_test.shape[0], Y_c_test.shape[1], 1))"""   

    ##################################### FEATURE LEVEL FUSION ############################################################
    elif scenario == 'FLF':        
        #print('Context interlocutor')

        if DB != 'SEM':
            # ADDING FEATURES OF INTERLOCUTOR
            interloc_features = np.zeros((features.shape[0],features.shape[1]))

            for i in range(0, interloc_features.shape[0]):
                ind = labels.index.values[i]

                splitted = list(ind)

                if splitted[3] == '1':
                    splitted[3] = '2'
                elif splitted[3] == '2':
                    splitted[3] = '1'

                ind = ''.join(splitted)

                interloc_features[i,:] = features.loc[ind].as_matrix()

            interloc_features = pd.DataFrame(interloc_features, index=features.index)

        elif DB == 'SEM':
            #print('Preparing SEMAINE')
            path = 'data/'
            interloc_features = pd.read_csv(path + 'SEM_features_LLD_10Hz_interlocutor_d.csv', index_col=0)
            interloc_features.index = np.array([x[:3] + '1' + x[4:] for x in interloc_features.index])


        features_train = features.loc[train_index]
        features_dev = features.loc[dev_index]
        features_test = features.loc[test_index]

        interloc_features_train = interloc_features.loc[train_index]
        interloc_features_dev = interloc_features.loc[dev_index]
        interloc_features_test = interloc_features.loc[test_index]

        scaler = preprocessing.StandardScaler()
        scaler_interloc = preprocessing.StandardScaler()

        # Scaling original features
        features_norm = scaler.fit_transform(features_train.as_matrix())

        features_norm = pd.DataFrame(features_norm, index=features_train.index)
        features_norm_dev = pd.DataFrame(scaler.transform(features_dev.as_matrix()), index=features_dev.index)
        features_norm_test = pd.DataFrame(scaler.transform(features_test.as_matrix()), index=features_test.index)

        # Scaling features of interlocutor
        interloc_features_norm = scaler.fit_transform(interloc_features_train.as_matrix())

        interloc_features_norm = pd.DataFrame(interloc_features_norm, index=interloc_features_train.index)
        interloc_features_norm_dev = pd.DataFrame(scaler.transform(interloc_features_dev.as_matrix()), 
                                                   index=interloc_features_dev.index)
        interloc_features_norm_test = pd.DataFrame(scaler.transform(interloc_features_test.as_matrix()), 
                                                   index=interloc_features_test.index)

        # Labels are common for original and interlocutor's features
        labels_train = labels.loc[train_index]
        labels_dev = labels.loc[dev_index] 
        labels_test = labels.loc[test_index] 

        #print('Subsampled')

        #features = features.merge(pd.DataFrame(interloc_features, index=labels.index), left_index=True, right_index=True).dropna()


        #print('Contextual preprocessing')

        [X_o_c, Y_c, I_c] = add_timesteps(features_norm.as_matrix(), labels.loc[train_index].as_matrix(), recordings.loc[train_index].as_matrix(),
                                        features_norm.index, timewindow, timewindow, sparsing, 0)

        [X_o_c_dev, Y_c_dev, I_c_dev] = add_timesteps(features_norm_dev.as_matrix(), labels.loc[dev_index].as_matrix(), recordings.loc[dev_index].as_matrix(),
                                        features_norm_dev.index, timewindow, timewindow, sparsing, 0)
        
        [X_o_c_test, Y_c_test, I_c_test] = add_timesteps(features_norm_test.as_matrix(), labels.loc[test_index].as_matrix(), recordings.loc[test_index].as_matrix(),
                                        features_norm_test.index, timewindow, timewindow, sparsing, 0)

        #print('Contextual preprocessing interlocutor')
        [X_i_c, __, __] = add_timesteps(interloc_features_norm.as_matrix(), labels.loc[train_index].as_matrix(), recordings.loc[train_index].as_matrix(),
                                        interloc_features_norm.index, timewindow, timewindow, sparsing_interlocutor, 0)

        [X_i_c_dev, __, __] = add_timesteps(interloc_features_norm_dev.as_matrix(), labels.loc[dev_index].as_matrix(), recordings.loc[dev_index].as_matrix(),
                                        interloc_features_norm_dev.index, timewindow, timewindow, sparsing_interlocutor, 0)
        
        [X_i_c_test, __, __] = add_timesteps(interloc_features_norm_test.as_matrix(), labels.loc[test_index].as_matrix(), recordings.loc[test_index].as_matrix(),
                                        interloc_features_norm_test.index, timewindow, timewindow, sparsing_interlocutor, 0)
        
        
        """Y_c = np.reshape(Y_c, (Y_c.shape[0], Y_c.shape[1], 1))
        Y_c_dev = np.reshape(Y_c_dev, (Y_c_dev.shape[0], Y_c_dev.shape[1], 1))
        Y_c_test = np.reshape(Y_c_test, (Y_c_test.shape[0], Y_c_test.shape[1], 1))"""
        
        

        #print('Concatenate')
        X_c = np.concatenate((X_o_c, X_i_c), axis=2)
        X_c_dev = np.concatenate((X_o_c_dev, X_i_c_dev), axis=2)
        X_c_test = np.concatenate((X_o_c_test, X_i_c_test), axis=2)

    return X_c, Y_c, I_c, X_c_dev, Y_c_dev, I_c_dev, X_c_test, Y_c_test, I_c_test

epoch_num = 10

timewindow = 10
[features, labels, recordings] = load_data(DB)


# Extracting recording numbers
recordings_num = np.unique([x[4:8] for x in np.unique(recordings)])
seed(RS)
np.random.shuffle(recordings_num)

# Defining model name
if SN != 'FLF':
    m_name = DB + '_' + ED + '_' + SN
else:
    m_name = DB + '_' + ED + '_' + SN + '_' + str(sparsing_interlocutor)

n_fold = 5
fold_size = int(np.ceil(len(recordings_num) / n_fold))

full_results = np.zeros(n_fold*(n_fold-1))

folds = [None] * n_fold

for f in range(n_fold):
    folds[f] = recordings_num[f*fold_size:np.min([(f+1)*fold_size,len(recordings_num)])]

count = 0
# Nested n-fold cross validation
for f_t in range(n_fold):
    # Fixing test fold
    test_fold = folds[f_t]
    #sys.stdout.write('Test fold: '  + str(test_fold))
    print('Test fold: '  + str(test_fold))
    
    for f_d in range(n_fold):
        if not f_d == f_t:
            # Checking different development fold, while keeping test the same
            dev_fold = folds[f_d]
            #sys.stdout.write('\tDev fold: '  + str(dev_fold))
            print('\tDev fold: '  + str(dev_fold))
            test_rec_num = set(test_fold)
            dev_rec_num = set(dev_fold)
            train_rec_num = set(recordings_num) - test_rec_num - dev_rec_num
            
            # Getting recording codes according to DBNC###, where "DBN" is database name, "C" - channel (1 or 2), "###" - recording number   
            train_rec = [DB + '1' + str(x) for x in train_rec_num]
            train_rec = train_rec + [DB + '2' + x for x in train_rec_num]

            dev_rec = [DB + '1' + str(x) for x in dev_rec_num]
            dev_rec = dev_rec + [DB + '2' + x for x in dev_rec_num]

            test_rec = [DB + '1' + str(x) for x in test_rec_num]
            test_rec = test_rec + [DB + '2' + x for x in test_rec_num]
        
            # ... and their indices
            train_index = get_index_by_code(recordings, train_rec)
            dev_index = get_index_by_code(recordings, dev_rec)
            test_index = get_index_by_code(recordings, test_rec)

            # Preprocess data
            [X_c,      Y_c,      I_c, 
             X_c_dev,  Y_c_dev,  I_c_dev, 
             X_c_test, Y_c_test, I_c_test] = preprocess_features(SN, features, labels, recordings, 
                                                                 train_index, dev_index, test_index, 
                                                                DB, sparsing_interlocutor=6)
            
            # Modeling
            CCC_dev_best = 0
            model = create_model(X_c, timewindow)
            for e in range(0,10):
                model.fit(X_c, Y_c, batch_size=128, epochs = 1, shuffle=True, verbose=0)

                # Evaluate development
                p = predict(model, X_c_dev, I_c_dev, dev_index)
                p_CCC_d = CCC_metric(p, labels.loc[dev_index].iloc[:,0].as_matrix())
                print('\t\tDevel: ' + str(p_CCC_d))
                
                if p_CCC_d > CCC_dev_best:
                    save_model(model, results_folder + '/' + m_name + '.hdf5')
                    CCC_dev_best = p_CCC_d

            best_model = load_model(results_folder + '/' + m_name + '.hdf5',compile=False)
            p = predict(best_model, X_c_test, I_c_test, test_index)

            p_CCC_t = CCC_metric(p, labels.loc[test_index].iloc[:,0].as_matrix())
            
            print('\t\t\tTest prediction: ' + str(p_CCC_t))
            
            full_results[count] = p_CCC_t
            count += 1
            
            np.save(results_folder + '/' + m_name, full_results)
            
            
