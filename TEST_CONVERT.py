
# coding: utf-8

# In[60]:


#!/usr/bin/env python3
import pandas as pd
import lz4.frame
import gzip
import io
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from glob import glob
from plumbum.cmd import rm
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras import regularizers
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os.path
import pickle
import datetime
import re
from keras.models import model_from_json


# In[61]:


def plotline(data):
    plt.figure()
    plt.plot(data)
    plt.legend()
    plt.show()

def event_count(time_series, data_name):
    time_series = time_series[['Fill Price (USD)']].values
    upevents = 0
    downevents = 0
    sameprice = 0
    prev_obv = time_series[0]
    for obv in time_series[1:]:
        if obv > prev_obv:
            upevents += 1
        elif obv < prev_obv:
            downevents += 1
        elif obv == prev_obv:
            sameprice += 1
        prev_obv = obv
    print('=== Event counts on %s ===' % data_name)
    print('upevents')
    print(upevents)
    print('downevents')
    print(downevents)
    print('sameprice')
    print(sameprice)
    print()

def mse(time_series, data_name):
    time_series = time_series[['Fill Price (USD)']].values
    total_squared_error = 0
    total_absolute_error = 0
    prev_obv = time_series[0]
    for obv in time_series[1:]:
        total_squared_error += (obv - prev_obv)**2
        total_absolute_error += abs(obv - prev_obv)
        prev_obv = obv
    num_predictions = len(time_series) - 1
    mean_squared_error = total_squared_error / num_predictions
    mean_absolute_error = total_absolute_error / num_predictions
    root_mean_squared_error = np.sqrt(mean_squared_error)
    print('=== baseline on %s ===' % data_name)
    print('total squared error')
    print(total_squared_error)
    print('total absolute error')
    print(total_absolute_error)
    print('mean squared error')
    print(mean_squared_error)
    print('mean absolute error')
    print(mean_absolute_error) 
    print('root mean squared error')
    print(root_mean_squared_error) 
    print()
    
def show_summary_statistics():
    #event_count(small_set, 'small')
    train_set = df.iloc[0:num_samples_training]
    dev_set = df.iloc[num_samples_training:num_samples_training+num_samples_dev]
    test_set = df.iloc[num_samples_training+num_samples_dev:]
    event_count(train_set, 'train')
    event_count(dev_set, 'dev')
    event_count(test_set, 'test')
    mse(train_set, 'train')
    mse(dev_set, 'dev')
    mse(test_set, 'test')
#show_summary_statistics()


# In[62]:


def initialize_model(X_train, loss, optimizer, num_LSTMs, num_units, dropout, predict_end_of_window):
    
    LSTM_input_shape = [X_train.shape[1], X_train.shape[2]]

    # DEFINE MODEL
    model = Sequential()

    if num_LSTMs == 2:
            model.add(LSTM(num_units[0], input_shape=LSTM_input_shape, return_sequences=True))
            model.add(Dropout(dropout))
            
            if predict_end_of_window:
                model.add(LSTM(num_units[1], return_sequences=False))
            else:
                model.add(LSTM(num_units[1], return_sequences=True))
        
    if num_LSTMs == 3:
            model.add(LSTM(num_units[0], input_shape=LSTM_input_shape, return_sequences=True))
            model.add(Dropout(dropout))

            model.add(LSTM(num_units[1], return_sequences=True))
            model.add(Dropout(dropout))
            
            if predict_end_of_window:
                model.add(LSTM(num_units[2], return_sequences=False))
            else:
                model.add(LSTM(num_units[2], return_sequences=True))

    if predict_end_of_window:
        model.add(Dense(1))
    else:
        model.add(TimeDistributed(Dense(1)))
      
    model.add(Activation('linear'))
    
    
    model.compile(loss=loss, optimizer=optimizer)
    
    return model


# In[63]:


def split_X(df, temporal_features, percent_train):
    n_all = df.shape[0]
    n_train = round(n_all * percent_train)
    n_dev   = round(n_all * ((1 - percent_train)/2))
    n_test  = round(n_all * ((1 - percent_train)/2))
    print('n_all:  ', n_all)
    print('n_train:', n_train)
    print('n_dev:  ', n_dev)
    print('n_test: ', n_test)
    
    if temporal_features:
        end = 59
    else:
        end = 16

    X_train = df.iloc[:n_train, 1:end].values.astype('float32')
    X_dev   = df.iloc[n_train:n_train+n_dev, 1:end].values.astype('float32')
    X_test  = df.iloc[n_train+n_dev:, 1:end].values.astype('float32')
    print(X_train.shape)
    print(X_dev.shape)
    print(X_test.shape)

    return X_train, X_dev, X_test


# In[64]:


def split_Y(df, percent_train):
    n_all = df.shape[0]
    n_train = round(n_all * percent_train)
    n_dev   = round(n_all * ((1 - percent_train)/2))
    n_test  = round(n_all * ((1 - percent_train)/2))
    
    Y_train = df.iloc[:n_train, -1:].values.astype('float32')
    Y_dev   = df.iloc[n_train:n_train+n_dev, -1:].values.astype('float32')
    Y_test  = df.iloc[n_train+n_dev:, -1:].values.astype('float32')
    print(Y_train.shape)
    print(Y_dev.shape)
    print(Y_test.shape)
    
    return Y_train, Y_dev, Y_test


# In[65]:


def df_to_parquet(df, outfile):
    pq.write_table(pa.Table.from_pandas(df), outfile, compression='snappy')


# In[66]:


def direction_prediction(y_true, y_pred, predict_end_of_window):
    if predict_end_of_window:
        prop_correct = np.sum(np.sign(y_pred) == np.sign(y_true)) / y_true.shape[0]
    else:
        prop_correct = np.sum(np.sign(y_pred) == np.sign(y_true)) / (y_true.shape[0] * y_true.shape[1])
    return prop_correct


# In[67]:


def evaluate_model(model, history, X_train, X_dev, X_test, Y_train, Y_dev, Y_test, predict_end_of_window):
    train_loss = history.history['loss'][-1]
    dev_loss   = history.history['val_loss'][-1]
    
    print('Evaluating test loss...')
    test_loss  = model.evaluate(X_test, Y_test, verbose=0)
    
    print('Predicting y_hat_train...')
    y_hat_train = model.predict(X_train)
    print('Predicting y_hat_dev...')
    y_hat_dev   = model.predict(X_dev)
    print('Predicting y_hat_test...')
    y_hat_test  = model.predict(X_test)
    
    train_prop_correct = direction_prediction(Y_train, y_hat_train, predict_end_of_window)
    dev_prop_correct   = direction_prediction(Y_dev, y_hat_dev, predict_end_of_window)
    test_prop_correct  = direction_prediction(Y_test, y_hat_test, predict_end_of_window)
    
    evaluation = {'train_loss': train_loss,
                  'dev_loss': dev_loss,
                  'test_loss': test_loss,
                  'train_prop_correct': train_prop_correct,
                  'dev_prop_correct': dev_prop_correct,
                  'test_prop_correct': test_prop_correct,
                  'y_hat_train': y_hat_train,
                  'y_hat_dev': y_hat_dev,
                  'y_hat_test': y_hat_test}

    return evaluation


# In[68]:


def create_sequenced_data(data, window, step):
    sequenced = []
    for minute in range(0, len(data) - window + 1, step):
        chunk = data[minute:minute+window]
        sequenced.append(chunk)
    sequenced = np.array(sequenced)
    return sequenced


# In[69]:


def split(df, temporal_features, percent_train):
    X_train, X_dev, X_test = split_X(df, temporal_features, percent_train)
    Y_train, Y_dev, Y_test = split_Y(df, percent_train)
    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test 


# In[70]:


def load_data(path, day_start=None, day_end=None):

    # Concatenate dataframes
    files = sorted(glob('%s/*.parquet' % path))
    
    if day_start is not None:
        start = day_start
    else:
        start = 0
    if day_end is not None:
        end = day_start
    else:
        end = len(files)
    files = files[start:end]
    
    all_dataframes = []
    for file in files:
        df = pq.read_table(file).to_pandas()
        all_dataframes.append(df)
    df = pd.concat(all_dataframes)
    return df


# In[71]:


def create_end_of_window_Y(Y, window, step):
    # Returns label at the end of a given window
    return np.array([Y[i] for i in range(window-1, len(Y), step)])


# In[72]:


def create_all_XY(X_train, X_dev, X_test, Y_train, Y_dev, Y_test, window_size, step, predict_end_of_window):
    X_train = create_sequenced_data(X_train, window=window_size, step=step)
    X_dev   = create_sequenced_data(X_dev,   window=window_size, step=step)
    X_test  = create_sequenced_data(X_test,  window=window_size, step=step)

    if predict_end_of_window:
        Y_train = create_end_of_window_Y(Y_train, window=window_size, step=step)
        Y_dev   = create_end_of_window_Y(Y_dev,   window=window_size, step=step)
        Y_test  = create_end_of_window_Y(Y_test,  window=window_size, step=step)
    else:
        Y_train = create_sequenced_data(Y_train, window=window_size, step=step)
        Y_dev   = create_sequenced_data(Y_dev,   window=window_size, step=step)
        Y_test  = create_sequenced_data(Y_test,  window=window_size, step=step)
    
    print('Train, dev, test shapes:')
    print(X_train.shape)
    print(X_dev.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_dev.shape)
    print(Y_test.shape)
    
    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test 


# In[73]:


def save_model_history(model, history, model_path):
  
    # serialize model to JSON
    if os.path.exists(model_path):
        suffix = ''.join(re.findall(r'\d+', str(datetime.datetime.now())))
        model_path = model_path + '_' + suffix

    os.makedirs(model_path)

    model_json = model.to_json()   
    with open(model_path + '/model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(model_path + '/model.h5')

    with open(model_path + '/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
        
    print("Saved model and history to:\n%s" % model_path)


# In[74]:


def plot_price(df, X_train, X_dev, field):
    X_train_stop = len(X_train)
    X_dev_stop = X_train_stop + len(X_dev)

    plt.figure(figsize=(20,4))
    plt.plot(np.arange(0, X_train_stop), df.iloc[0:X_train_stop][field], 'k')
    plt.plot(np.arange(X_train_stop, X_dev_stop), df.iloc[X_train_stop:X_dev_stop][field], 'r')
    plt.plot(np.arange(X_dev_stop, len(df)), df.iloc[X_dev_stop:len(df)][field], 'g')


# In[75]:


def plot_train_dev_losses(history):
    train_loss = history.history['loss']
    dev_loss   = history.history['val_loss']
    
    plt.figure(figsize=(20,4))
    plt.plot(np.log(train_loss), 'k')
    
    plt.figure(figsize=(20,4))
    plt.plot(np.log(dev_loss), 'b')
    
    plt.figure(figsize=(20,4))
    plt.plot(train_loss, 'k')
    
    plt.figure(figsize=(20,4))
    plt.plot(dev_loss, 'b')
    
    plt.show()


# In[76]:


def plot_percent_change(y_pred, y_true, timestep_within_window, minute_start, minute_end, predict_end_of_window):
    
    ys=[]
    for i in range(len(y_pred)):
        ys.append(y_pred[i][timestep_within_window])

    original_ys=[]
    for i in range(len(y_true)):
        original_ys.append(y_true[i][timestep_within_window])

        
    ys_orig = np.array(original_ys)
    ys_pred = np.array(ys)
    
    
    OldRange = (ys_pred.max() - ys_pred.min())  
    NewRange = (ys_orig.max() - ys_orig.min())   
    new_ys_pred = (((ys - ys_pred.min()) * NewRange) / OldRange) + ys_orig.min()
    
    
    norm1 = ys_orig / np.linalg.norm(ys_orig)
    norm2 = ys_pred / np.linalg.norm(ys_pred)
    

    plt.figure(figsize=(20,10))
    plt.plot(norm1[minute_start:minute_end], 'k', alpha=0.9)
    plt.plot(norm2[minute_start:minute_end], 'r', alpha=0.9)
    
    plt.figure(figsize=(20,10))
    plt.plot(original_ys[minute_start:minute_end], 'k', alpha=0.9)
    plt.plot(ys[minute_start:minute_end], 'r', alpha=0.9)
    
    plt.figure(figsize=(20,10))
    plt.plot(original_ys[minute_start:minute_end], 'k', alpha=0.9)
    plt.plot(new_ys_pred[minute_start:minute_end], 'r', alpha=0.9)
    
    
    try:
        plt.figure(figsize=(20,10))
        plt.plot(y_true[minute_start:minute_end], 'k', alpha=0.9)
        plt.plot(y_pred[minute_start:minute_end], 'r', alpha=0.9)
    except:
        a = 1


# In[77]:


def print_save_events_props(train, dev, test, evaluate, model_name, model_path):
    
    train_event_counts = np.unique(np.sign(train), return_counts=True)
    train_event_prop   = train_event_counts[1] / len(train)
    
    dev_event_counts = np.unique(np.sign(dev), return_counts=True)
    dev_event_prop   = dev_event_counts[1] / len(dev)
    
    test_event_counts = np.unique(np.sign(test), return_counts=True)
    test_event_prop   = test_event_counts[1] / len(test)
    
    print(model_name)
    print('\n========== EVENT COUNTS AND PROPORTIONS ==========')
    print('=== TRAIN ===')
    print('Down, Same, Up:', train_event_counts[1])
    print('Down, Same, Up:', train_event_prop)
    
    print('\n=== DEV ===')
    print('Down, Same, Up:', dev_event_counts[1])
    print('Down, Same, Up:', dev_event_prop)
    
    print('\n=== TEST ===')
    print('Down, Same, Up:', test_event_counts[1])
    print('Down, Same, Up:', test_event_prop)
    
    
    print('\n========== CORRECTION DIRECTION PREDICTIONS ==========')
    print("TRAIN: %f\nDEV:   %f\nTEST:  %f" % (evaluate['train_prop_correct'], 
                                               evaluate['dev_prop_correct'], 
                                               evaluate['test_prop_correct']))
    
    print('\n========== FINAL LOSS ==========')
    print("TRAIN: %s\nDEV:   %s\nTEST:  %s\n" % (evaluate['train_loss'], evaluate['dev_loss'], evaluate['test_loss']))   


# In[78]:


def restore_model(path):
  return model


# In[79]:


def restore_history(path):
  return history


# In[80]:


def plot_losses(history, field, title):
  vals = np.log(restored_history[field])
  #vals = restored_history[field]
  new_df = pd.DataFrame(vals, columns=[field])
  new_df.plot(y = field, figsize=(7,6), title=title, fontsize=14, legend=False, color='firebrick')
  plt.xlabel('Epoch', fontsize=18)
  plt.title(title, fontsize=15, fontweight='bold')


# In[81]:


def mse(time_series):
    total_squared_error = 0
    total_absolute_error = 0
    prev_obv = time_series[0]
    for obv in time_series[1:]:
        total_squared_error += (obv - prev_obv)**2
        total_absolute_error += abs(obv - prev_obv)
        prev_obv = obv
    num_predictions = len(time_series) - 1
    mean_squared_error = total_squared_error / num_predictions
    mean_absolute_error = total_absolute_error / num_predictions
    root_mean_squared_error = np.sqrt(mean_squared_error)
    print('=== baseline ===')
    print('total squared error')
    print(total_squared_error)
    print('total absolute error')
    print(total_absolute_error)
    print('mean squared error')
    print(mean_squared_error)
    print('mean absolute error')
    print(mean_absolute_error) 
    print('root mean squared error')
    print(root_mean_squared_error) 
    print()


# In[151]:


##### MAIN MODEL #####

# HYPERPARAMETERS
window_size = 60
step = 1
predict_end_of_window = False
temporal_features = False

batch_size = 2048 #8192
num_epochs = 30
verbose = 1
loss = 'mean_squared_error'
optimizer = 'adam'
num_LSTM = 3
num_units = [128, 256, 256]
#num_LSTM = 2
#num_units = [256, 256]
dropout = 0.1

path = 'cboe/parquet_preprocessed_BTCUSD_merged'
day_start = 401
day_end = None
percent_train = 0.9

num_units_string = '-'.join([str(u) for u in num_units])

model_name = 'window-%s_step-%s_predEndWindow-%s_temporalFeat-%s_batch-%s_epochs-%s_loss-%s_opt-%s_numLSTMs-%s_numUnits-%s_dropout-%s_dayStart-%s_dayEnd-%s' % (window_size, step, str(predict_end_of_window), str(temporal_features), batch_size, num_epochs, loss, optimizer, num_LSTM, num_units_string, dropout, str(day_start), str(day_end))
print(model_name+'\n')
model_path = 'models/%s' % model_name  

early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=15)
callbacks_list = [early_stop]

# LOAD DATA
df = load_data(path, day_start, day_end)

# CREATE XY DATA
X_train, X_dev, X_test, Y_train, Y_dev, Y_test = split(df, temporal_features, percent_train)
X_train, X_dev, X_test, Y_train, Y_dev, Y_test = create_all_XY(X_train, X_dev, X_test, Y_train, Y_dev, Y_test,
                                                               window_size, step, predict_end_of_window)
plot_price(df, X_train, X_dev, field='current_price')
plot_price(df, X_train, X_dev, field='percent_change')


# In[ ]:


# INITIALIZE MODEL
model = initialize_model(X_train, loss, optimizer, num_LSTM, num_units, dropout, predict_end_of_window=predict_end_of_window)

# TRAIN MODEL
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs, 
                    validation_data=(X_dev, Y_dev), callbacks=callbacks_list, verbose=verbose, shuffle=True) 

# SAVE MODEL AND HISTORY
save_model_history(model, history, model_path)


# In[ ]:


# EVALUATE MODEL
print('Evaluating...')
evaluate = evaluate_model(model, history, X_train, X_dev, X_test, Y_train, Y_dev, Y_test, predict_end_of_window)


# In[ ]:


# VISUALIZE
## Plot: 
# Historical price, color coded with train, dev, test
# Historical percent change, color coded with train, dev, test
# Train Loss, Dev Loss
# Actual price vs predicted price (or percent change) for test set
# Example features time series for one day (NOTE: in the preprocessing_final notebook)

if predict_end_of_window:
    timestep_within_window = 0
else:
    timestep_within_window = window-1
    
minute_start = 0

print_save_events_props(Y_train.flatten(), Y_dev.flatten(), Y_test.flatten(), evaluate, model_name, model_path)
plot_price(df, X_train, X_dev, field='current_price')
plot_price(df, X_train, X_dev, field='percent_change')

plot_losses(restored_history, field='loss', title='Training Loss')
plot_losses(restored_history, field='val_loss', title='Dev Loss')

minute_end = len(Y_train)
plot_percent_change(evaluate['y_hat_train'], Y_train, timestep_within_window, minute_start, minute_end, predict_end_of_window)

minute_end = len(Y_dev)
plot_percent_change(evaluate['y_hat_dev'], Y_dev, timestep_within_window, minute_start, minute_end, predict_end_of_window)

minute_end = len(Y_test)
plot_percent_change(evaluate['y_hat_test'], Y_test, timestep_within_window, minute_start, minute_end, predict_end_of_window)


# In[54]:


paths = sorted(glob('models/*'))
file = 'models/window-60_step-1_predEndWindow-False_temporalFeat-True_batch-4096_epochs-30_loss-mean_squared_error_opt-adam_numLSTMs-3_numUnits-128-256-256_dropout-0.1_dayStart-401_dayEnd-None'
restored_history = pickle.load( open( file+"/trainHistoryDict", "rb" ) )
plot_losses(restored_history, field='loss', title='Training Loss')
plot_losses(restored_history, field='val_loss', title='Dev Loss')


# In[140]:


def mse_rmse(Y_true, Y_pred, baseline_desc):
  MSE = np.sum((Y_true - Y_pred) ** 2) / len(Y_true)
  RMSE = np.sqrt(MSE)
  print('\n%s' % baseline_desc)
  print('MSE:  %06f' % MSE)
  print('RMSE: %06f' % RMSE)


# In[141]:


def baselines(Y, data_name):
  print('\n========= %s =========' % data_name)
  Y_true = np.array(Y[1:])
  
  # Predict no percent change
  Y_pred = np.zeros((Y_true.shape))
  mse_rmse(Y_true, Y_pred, 'Predict NO Percent Change')
  
  # Predict same percent change
  Y_pred = np.array(Y[:-1])
  mse_rmse(Y_true, Y_pred, 'Predict SAME Percent Change')


# In[142]:


print('================== BASELINES ==================')
baselines(Y_train, data_name='Training Data')
baselines(Y_dev,   data_name='Dev Data')
baselines(Y_test,  data_name='Test Data')


# In[66]:


#per row: train, dev, test loss
losss = np.array([[1.690e-6, 5.907e-6, 5.893e-6],
[1.673e-6, 5.79e-6, 5.865e-6],
[1.925e-6, 1.043e-5, 1.18e-6],
[1.959e-6, 5.909e-6, 5.97e-6],
[1.694e-6, 6.255e-6, 6.31e-6],
[1.799e-6, 5.791e-6, 5.94e-6],
[1.822e-6, 1.00e-5, 1.09e-5],
[1.718e-5, 6.934e-5, 6.988e-5]])


# In[145]:


np.sqrt(np.array([1.673e-6, 5.79e-6, 5.865e-6]))


# In[67]:


np.sqrt(losss.mean(axis=0, keepdims=True))

