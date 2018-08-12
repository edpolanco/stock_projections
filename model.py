from data_manager import MultiSequence, split_data,Downloader,Feature_Selection
import numpy as np
import pandas as pd
import os
import sys
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.optimizers import RMSprop
from keras.models import model_from_json
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class Predictor(object):
    """
    Class for predicting daily future stock prices.

    Parameters
    ----------
    symbol : str
        Company trade symbol.
    """
    def __init__(self, symbol: str):
        self.__symbol = symbol


        file_loc = "./data/{0:}/normalized.csv".format(self.symbol)
        if os.path.isfile(file_loc):
            data = pd.read_csv(file_loc)
            self.__closing_prices = data["close"].values
            self.__max_price = max(self.__closing_prices)
            self.__min_price = min(self.__closing_prices)
        else:
            print("File does not exist:",file_loc)

    @property
    def symbol(self):
        """ Get the company trade symbol """ 
        return self.__symbol
    
    @staticmethod
    def download_prep(symbol: str, start_date: str, end_date: str):
        """
        Download historical stock prices from yahoo finance
        and then prepare the data for neural network input
        to then make predictions from our trained models.  

        Parameters
        ----------
        symbol : str
            Company trade symbol.

        start_date : str 
            The starting trade date in 'YYYYYMMDD' format.
    
        end_date : str 
            The ending trade date in 'YYYYYMMDD' format.    
        """
        #download data and save 
        download = Downloader(symbol,start_date, end_date)
        download.save()

        #prep data for NN input
        #check if file exist first
        file_path = "./data/{}/quotes.csv"
        if os.path.isfile(file_path.format(symbol)):
            feature = Feature_Selection.read_csv(symbol, file_path.format(symbol))
            feature.calculate_features()
            feature.normalize_data()
            feature.save_stock_data()
            feature.save_normalized_data()
        else:
            print("File does not exist:",file_path.format(symbol))

    def select_model(self, verbose = 0):
        """
        Selects best model from all the trained models saved in the model directory
        and then stores the later for later predctions.
        
        Parameters
        ------------------
        verbose: int [defualt 0]
            print to screen indicator.  
            
            If 0 does not print anything to screen.

            If 1 prints symbol and test error for each model tested.

            If 2 prints the stock symbol of the model selected.
        """

        #first get list of the available models
        tickers =  [x.split('\\')[1] for x,_,_ in os.walk(ModelLoader.root_path()) if len(x.split('\\')) > 1  ]
        
        #now let find the best model
        best_model = None
        lowest_test_error = 2.0
        
        #prepare our sequence data
        for idx, ticker in enumerate(tickers,1):
            try:                
                loaded_model = ModelLoader(ticker)
                seq_obj = MultiSequence(self.symbol,loaded_model.window_size,1)
                testing_error =loaded_model.model.evaluate(seq_obj.X,seq_obj.y, verbose=0)
                
                if verbose == 1:
                    print(">{0:>3}) Now checking model: {1:<5}  Test error result: {2:.4f}".format(idx,ticker, testing_error))
                
                if lowest_test_error > testing_error:
                    best_model = loaded_model
                    lowest_test_error = testing_error
            except :
                pass
            
        #save best model
        self.__best_model = best_model
        self.__test_error = lowest_test_error
        if verbose in[1,2]:
            print("==> Best model ticker {0:} with error of {1:.4f}".format(self.__best_model.ticker,self.__test_error))

    def graph(self):
        """ Graphs the stock predictions from choosen model vs. the actual
        """
        seq_obj = MultiSequence(self.symbol, self.__best_model.window_size,1)
        test_predict = self.__best_model.model.predict(seq_obj.X)

        #our data is scaled between -1 and 1 so lets scale it back up
        scaler = MinMaxScaler(feature_range=(self.__min_price ,self.__max_price))
        orig_data = seq_obj.original_data.reshape(-1,1)
        orig_prices = scaler.fit_transform(orig_data).flatten()
        
        # plot actual prices
        plt.plot(orig_prices, color='k')
    
        # plot test set prediction after scaling back up
        length = len(seq_obj.X) + self.__best_model.window_size 
        test_in = np.arange(self.__best_model.window_size,length,1)
        pred_prices = scaler.fit_transform(test_predict.reshape(-1,1)).flatten()
        plt.plot(test_in,pred_prices,color = 'b')
        
        # pretty up graph
        plt.xlabel('day')
        plt.ylabel('Closing price of stock')
        plt.title("Price prediction for {}".format(self.symbol))
        plt.legend(['Actual','Prediction'],loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

def final_model(X:np.array,y:np.array, learn_rate:float, dropout:float):
    """
    *** final_model() ***
    ========================
        RNN model with following layers:
        1) Bidirectional LSTM layer (output size based on X input sequence length)
        2) Fully connected layer (output based on input sequence length)
        3) Dropout (based on given dropout rate) 
        4) Fully connected tanh output layer of 1

    Parameters
    -----------
    X: numpy array
        input sequence data.

    y: numpy array
        target sequence data.

    learn_rate: float
        Neural network learning rate.
        
    dropout: float
        Dropout rate.

    Returns 
    -----------
    Returns compiled keras sequential model
    """
    model = Sequential()
    model.add(Bidirectional(LSTM(X.shape[1],return_sequences=False), input_shape=(X.shape[1:])))
    model.add(Dense(X.shape[1]))
    model.add(Dropout(dropout))
    model.add(Dense(y.shape[1], activation='tanh'))

    # compile the model
    optimizer = RMSprop(lr=learn_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def model_selector(ticker:str, window_sizes:list, learn_rates:list, dropouts:list, epochs:list, batch_size:int,verbose=0):
    """
    *** model_selector() ***
    ========================

        Selects the best peforming (base on lowest MSE on test data) model for a given ticker based on 
        the given widow sizes, learn rates, drop out rates, and epoch counts

    Parameter
    -----------
    ticker: str
        company trade symbol.

    window_size: list
        List of sequence lengths to test.

    learn_rates: list
        List of learning rates to test.
        
    dropouts: list`
        List of dropouts rates to test.

    epochs: list
        List of epochs to test.
    
    batch_size: int
        training batch_size

    verbose: int [defualt 0]
        print to screen indicator.  
        If 0 does not print anything to screen.

        If 1 prints detail for each model tested and also the 
        summary detail for choosen model to screen.

        If 2 prints summary detail for choosen model to screen.

    Returns
    -----------
    Tuple with two items.  First item is the keras trained model and 
    second a dictionary with the following keys:
        1) ticker: company trade symbol of the trained company 
        2) test_error :   the model MSE test error  
        3) train_error:  the moodel MSE training error 
        4) dropout:  the moodel drop out rate
        5) epoch:  the moodel epoch count
        6) learn_rate:  the moodel learning rate
        7) window_size:  the sequence length of the input data
    """
    #our best model variables
    best_model = None
    lowest_test_error = 2.0
    best_training_error =0.0
    best_learn_rate = 0.0
    best_dropout_rate = 0.0
    best_epoch = 0
    best_window_size = 0
    
    counter = 1

    if verbose == 1:
            print("*** Best Model Selection for {} ***".format(ticker))
            print("=" * 60)      
    
    for window_size in window_sizes:
        if verbose == 1:
            print("\nWindow size: {}".format(window_size))
            print('-' * 60)
        
        #prepare our sequence data
        seq_obj = MultiSequence(ticker,window_size,1)
        X_train,y_train,X_test,y_test = split_data(seq_obj)    
    
        for rate in learn_rates:
            for dropout in dropouts:
                for epoch in epochs:
                    model = final_model(X_train,y_train,rate,dropout)
                    model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, verbose=0)

                    # print out training and testing errors
                    training_error = model.evaluate(X_train, y_train, verbose=0)
                    testing_error = model.evaluate(X_test, y_test, verbose=0)
                    
                    if verbose == 1:
                        msg = " > Learn rate: {0:.4f} Dropout: {1:.2f}"
                        msg += " Epoch: {2:} Training error: {3:.4f} Testing error: {4:.4f}"
                        msg = "{0:2}".format(str(counter)) + "   " +msg.format(rate,dropout, epoch, training_error, testing_error)
                        print(msg)

                    #check if test error 
                    if lowest_test_error > testing_error:
                        best_model = model
                        lowest_test_error = testing_error
                        best_learn_rate = rate
                        best_dropout_rate = dropout
                        best_epoch = epoch
                        best_training_error = training_error 
                        best_window_size = window_size

                    #increase our print counter
                    counter += 1
    
    if verbose in [1,2]:
        print("\nModel selection summary for {} with window size of {}:".format(ticker,best_window_size))
        print('-' * 60)
        msg = " ==> Learn rate: {0:.4f} Dropout: {1:.2f}"
        msg += " Epoch: {2:} Training error: {3:.4f} Testing error: {4:.4f}"
        msg = msg.format(best_learn_rate,best_dropout_rate, best_epoch, best_training_error, lowest_test_error)
        print(msg)
                        
    best_dict ={}
    best_dict["ticker"] = ticker
    best_dict["test_error"] =  float("{0:.4f}".format(lowest_test_error) )  
    best_dict["learn_rate"] = best_learn_rate
    best_dict["dropout"] = best_dropout_rate
    best_dict["epoch"] = best_epoch
    best_dict["train_error"] =  float("{0:.4f}".format(best_training_error)  ) 
    best_dict["window_size"] = best_window_size
    
    

    return (best_model,best_dict)

class ModelLoader(object):
    """ Class for saving and loading keras models to and from
        directory.  
    """
    #static variables for saving and reading to and from disk
    __sub_folder = "./model/{0:}"
    __model_path = "./model/{0:}/{0:}_model.json"
    __weights_path = "./model/{0:}/{0:}_weights.h5"
    __prop_path = "./model/{0:}/{0:}_train_props.json"

    def __init__(self,symbol: str):
        try:
            #check if directory exist            
            if not os.path.isfile(ModelLoader.__model_path.format(symbol)):
                print("No model exist for {}".format(symbol))
                return

            if not os.path.isfile(ModelLoader.__weights_path.format(symbol)):
                print("No weigths file exist for {}".format(symbol))
                return
            
            if not os.path.isfile(ModelLoader.__prop_path.format(symbol)):
                print("No training properties file exist for {}".format(symbol))
                return            
            
            # load json and create model and then load weights
            with open(ModelLoader.__model_path.format(symbol), 'r') as json_file:
                loaded_model_json = json_file.read()
                #json_file.close()
                loaded_model = model_from_json(loaded_model_json)
                loaded_model.load_weights(ModelLoader.__weights_path.format(symbol))
            
                #compile model
                loaded_model.compile(loss='mean_squared_error', optimizer='rmsprop')
                self.__model = loaded_model
                
            #load model training properities
            with open(ModelLoader.__prop_path.format(symbol), 'r') as prop_file:
                self.__train_prop = json.load(prop_file)

        except OSError as err:
            print("OS error for symbol {}: {}".format(symbol, err))
        except:
            print("Unexpected error for symbol {}:{}".format(symbol, sys.exc_info()[0]))        


    @staticmethod
    def root_path():
        return "./model"

    @property
    def model(self):
        """ Get keras compiled RNN model """
        return self.__model

    @property
    def ticker(self):
        """Get trade symbol of the model """
        return self.__train_prop["ticker"]

    @property
    def window_size(self):
        """Get the sequence length of the input data """
        return self.__train_prop["window_size"]

    @property
    def train_prop(self):
        """ Get model training properties as dict.
        """
        return self.__train_prop
    
    @staticmethod
    def save(symbol: str,model:Sequential,train_props: dict):
        """ Save keras model to model directory
            Creates two files and one json file with the model properities 
            and another h5 file with medel weigths.

            Parameter
            -----------
            symbol: str
                    company trade symbol

            model: keras model

            train_props: dict
                        model training properities: 
                        - ticker
                        - test error
                        - train error
                        - learn rate
                        - dropout rate
                        - epoch
                        - window size
        """
        try:
            #create if it doesn't exist            
            if not os.path.isdir(ModelLoader.__sub_folder.format(symbol)):
                os.makedirs(ModelLoader.__sub_folder.format(symbol))

            #serialize keras model to json
            model_json = model.to_json()
            with open(ModelLoader.__model_path.format(symbol), "w") as json_file:
                json_file.write(model_json)
    
            # serialize weights to HDF5
            model.save_weights(ModelLoader.__weights_path.format(symbol))

            #save model training properities
            with open(ModelLoader.__prop_path.format(symbol), 'w') as prop_file:
                json.dump(train_props, prop_file)

        except OSError as err:
            print("OS error for symbol {}: {}".format(symbol, err))
        except:
            print("Unexpected error for symbol {}:{}".format(symbol, sys.exc_info()[0]))