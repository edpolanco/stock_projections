
from abc import ABC, abstractmethod,ABCMeta
import sys
import os
import math
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from yahoo_quote_download import yqd
from datetime import datetime
from collections import OrderedDict,Set
import numpy as np 
import matplotlib.pyplot as plt


def companies():
    """Import company listing from csv file and return as dataframe
        Dataframe has the following columns: 'Company','Symbol', 'Industry'
    """
    dataset = pd.read_csv(os.path.join("data", "dow30.csv"))
    return  dataset

def symbol_list():
    """Import company listing from csv file and return trade symbols as a list """
    dataset = pd.read_csv(os.path.join("data", "dow30.csv"))
    return  dataset['Symbol'].values.tolist()

class BaseData(object):
    """
    Base helper class for reading, writing and transformating stock quotes  data

    Parameters
    ----------
    symbol : str
        Company trade symbol.
    """

    def __init__(self, symbol: str):
        self.__symbol = symbol 

    @property
    def symbol(self):
        """ Get the company trade symbol """ 
        return self.__symbol

    def save(self,file_dir: str, file_name: str, data: pd.DataFrame):
        """ Save quote data to csv file

            Parameters
            ----------
            file_dir:  str
                Root file directory.
                Defualt value
            
            file_name: str
                file full path within root directory.

            data: DataFrame
                The Pandas DataFrame that we want to write to csv file.
            
        """
        #first check if directory for trade symbol exist
        try:
            #make sure data not none type
            if data is None:
                return 

            full_path  = os.path.join(file_dir,file_name)
            include_index = False if data.index.name == None else True
            if os.path.isdir(file_dir):
                data.to_csv(full_path, index = include_index)
            else:
                os.makedirs(file_dir)
                data.to_csv(full_path, index = include_index)
        except OSError as err:
            print("OS error for symbol {}: {}".format(self.symbol, err))
        except:
            print("Unexpected error for symbol {}:{}".format(self.symbol, sys.exc_info()[0]))

class Downloader(BaseData):
    """
    Downloads historical end of day stock quotes from yahoo for a given trade symbol 
    and a given date range.  The downloaded data is stored as a DataFrame.

    Parameters
    ----------
    symbol : str
        Company trade symbol.
    
    start_date : str 
        The starting trade date in 'YYYYYMMDD' format.
    
    end_date : str 
        The ending trade date in 'YYYYYMMDD' format.
    """
    def __init__(self, symbol: str, start_date: str, end_date: str):
        try:
            BaseData.__init__(self,symbol)
            # self.__symbol = symbol
            self.__start_date =  datetime.strptime(start_date, '%Y%m%d') 
            self.__end_date =datetime.strptime(end_date, '%Y%m%d') 
            self.__data = None
           
            #download yahoo data
            #returns a list where the first element is the header
            # example: ['Date,Open,High,Low,Close,Adj Close,Volume', '2017-05-15,55.080002,55.490002,55.080002,55.400002,53.346981,10686100', '']
            yah = yqd.load_yahoo_quote(symbol, start_date, end_date)

            #retrieve Date,Open,High,Low,Close,Adj Close,Volume as list
            header = yah[0].split(',')
            
            #loop through remaining elements and add to dictionary
            table = []
            for i  in yah[1:]:
                #make sure we have complete quote
                quote = i.split(',')
                
                if len(quote) > 1:
                    d = dict()
                    d[header[0]] = quote[0] # Date
                    d[header[1]] = float(quote[1]) # Open
                    d[header[2]] = float(quote[2]) # High
                    d[header[3]] = float(quote[3]) # Low
                    d[header[4]] = float(quote[4]) # close
                    d[header[5]] = float(quote[5]) # Adj Close
                    d[header[6]] = int(quote[6]) # Volume
                    table.append(d)

            #Create DataFrame
            self.__data = pd.DataFrame(table)
            self.__size = len(self.__data)
        except OSError as err:
            print("OS error for symbol {}: {}".format(symbol, err))

    def save(self):
        """ Save quote data to csv file
        """
        file_dir = os.path.join("./data",self.symbol)
        BaseData.save(self, file_dir ,"quotes.csv", self.__data)

    @property
    def start_date(self):
        """ Get the start quote date """ 
        return self.__start_date
    
    @property
    def end_date(self):
        """ Get the ending quote date """ 
        return self.__end_date

    @property
    def data(self):
        """ Get the trade data as a DataFrame """ 
        return self.__data

    @property
    def size(self):
        """ The number of quotes retreived from yahoo """ 
        return self.__size

class Feature_Selection(BaseData):
    """
        Takes DataFrame of yahoo trade data for a given trade symbol
        and performs transformations to creates features for neural network training.

        Parameters
        ----------
        symbol: str
            Company trade symbol
        data: Pandas DataFrame object
            DataFrame object must have the following columns:
                1. 'Date'
                2. 'Open'
                3. 'High'
                4. 'Low'
                5. 'Close'
                6. 'Adj Close'
                7. 'Volume'
            mfi_days: int
            The number of days for the Money Flow Index (MFI) calculation
    """    
    def __init__(self,symbol: str, data:pd.DataFrame, mfi_days = 14):
        
        BaseData.__init__(self,symbol)
        
        self.__days = mfi_days
        
        #dataframe data variables
        self.__data = None
        self.__data_normal = None

        #check columns
        cols = data.columns.values
        cols_check = "Date,Open,High,Low,Close,Adj Close,Volume".split(',')
        missing = False
        for col in cols:
            found = False
            for name in cols_check:
                if col == name:
                    found = True
                    break
            
            if not found:
                print("The column {} is missing.".format(col)) 
                missing = True
                break
        
        if not missing:
            self.__data = data
            self.__data['Date'] = pd.to_datetime(self.__data['Date'])
            
            # just to make sure dates are sorted and then re-index 
            self.__data.sort_values('Date',inplace=True) 
            self.__data.reset_index(drop=True,inplace=True) 
            self.__data.index.name = 'index'
 
    @classmethod
    def read_csv(cls,symbol: str, file_loc: str):
        """
        Reads a csv file of yahoo quotes and returns a Feature_Selection object
        
        Parameters
        ----------
        symbol : str
            Company trade symbol

        file_loc : str
            Directory location of csv file.
            The csv file must contain the  following columns:
                1. 'Date'
                2. 'Open'
                3. 'High'
                4. 'Low'
                5. 'Close'
                6. 'Adj Close'
                7. 'Volume'

        Returns
        -------
        returns Feature_Selection class

        """
        try:
            data = pd.read_csv(file_loc)
            return cls(symbol, data)
        except OSError as err:
            print("OS error {}".format(err))
            return None
         
    @property
    def data(self):
        """ Get the trade data as a DataFrame """ 
        return self.__data

    @property
    def data_normal(self):
        """ Get the trade data as a DataFrame """ 
        return self.__data_normal

    def calculate_features(self):
        """We select two features for Neural Network training:
            1.  Adjusted Close log returns
            2.  Money Flow Index (MFI)

            When we call this method we add a new column to store log returns.
            We then calculate the MFI and store it into a new column.  
         """
         #Calculates daily log returns on adj closing prices. 
        self.__cal_log_return("Adj Close")
        self.__cal_mfi()
    

    def __scale_data(self,col_Name:str):
        """Normalize a given column between (-1,1) and return it as a 1-dimensional array
            
            Parameters
            ----------
            col_name: str
                    The name of the column we want to normalize
        """
        values = self.__data[col_Name].iloc[self.__days:].values.reshape(-1,1)     
        scaler = MinMaxScaler(feature_range=(-1,1))
        return scaler.fit_transform(values).flatten()

    def __flatten_data(self,col_Name:str):
        """Get a 1-dimensional array for a given column that is
            complete considering the number of days used for mfi index calculation
            
            Parameters
            ----------
            col_name: str
                    The name of the column we want 
        """
        return self.__data[col_Name].iloc[self.__days:].values.flatten() 

    def normalize_data(self):
        """
            Normalize Adj Close log returns and the mfi index 
            between -1 and 1 for neural network training. 
            The normalized data is saved as a DataFrame. 
        """
        #get index so we can reference back
        index = self.__data.index.values[self.__days:]

        table =  OrderedDict()
        table['close'] = self.__flatten_data('Adj Close')
        table['returns'] = self.__flatten_data('Adj Close_log_returns')
        table['mfi'] =self.__flatten_data('mfi_index')
        table['normal_close'] = self.__scale_data('Adj Close')
        table['normal_returns'] = self.__scale_data('Adj Close_log_returns')
        table['normal_mfi'] = self.__scale_data('mfi_index')
        self.__data_normal = pd.DataFrame(table, index=index)
        self.__data_normal.index.name = 'index'
        
    def __cal_log_return(self,col_name: str):
        """ Creates a new column and calculates 
            daily log returns for given column name. 
        Parameters
            ----------
            col_name :  str
                        The name of the column we want to calculate the log returns.
        """ 
        values = self.__data[col_name].values
        log_returns = np.zeros_like(values)
        for idx in range(1,len(values)):
            log_returns[idx] = math.log( values[idx]/values[idx -1] ) 
        
        self.__data[col_name+"_log_returns"] = pd.Series(log_returns, index = self.__data.index)
    
    def save_stock_data(self):
        """ Save processed quote data to csv file
        """
        file_dir = os.path.join("./data",self.symbol)
        BaseData.save(self, file_dir ,"quote_processed.csv", self.__data)
            
    def save_normalized_data(self):
        """ Save normalized data to csv file
        """
        file_dir = os.path.join("./data",self.symbol)
        BaseData.save(self, file_dir ,"normalized.csv", self.__data_normal)

    def __cal_mfi(self):
        """
            Calculate the Money Flow Index (MFI) and 
            stores the data in a new colum
        """
        #cal typical price
        typ_price = pd.DataFrame((self.__data["High"] + self.__data["Low"] + self.__data["Adj Close"])/3, columns =["price"] )
        
        #add volume to dataframe
        typ_price['volume'] = self.__data["Volume"]
        
        #positive flow column
        typ_price['pos'] = 0
        
        #negative flow column
        typ_price['neg'] = 0

        #MFI index column as float type
        typ_price['mfi_index'] = 0.0

        #calculate positive and negative flow
        for idx in range(1,len(typ_price)):
            if typ_price['price'].iloc[idx] > typ_price['price'].iloc[idx-1]:
                #positive flow
                typ_price.at[idx,'pos' ] = typ_price['price'].iloc[idx] * typ_price['volume'].iloc[idx]     
                # typ_price.set_value(idx,'pos' ,typ_price['price'].iloc[idx] * typ_price['volume'].iloc[idx])     
            else:
                #negative flow
                typ_price.at[idx,'neg'] = typ_price['price'].iloc[idx] * typ_price['volume'].iloc[idx]
                # typ_price.set_value(idx,'neg' ,typ_price['price'].iloc[idx] * typ_price['volume'].iloc[idx])

        #calculate positive and negative flow
        pointer = 1
        for idx in range(self.__days,len(typ_price)):
            pos = typ_price['pos'].iloc[pointer:idx + 1].sum()
            neg = typ_price['neg'].iloc[pointer:idx + 1].sum()
            
            #make sure we don't divide by zero
            if neg != 0:
                base = (1.0 + (pos/neg))
            else:
                base = 1.0
            typ_price.at[idx,'mfi_index'] = 100.0 - (100.0/base )
            # typ_price.set_value(idx,'mfi_index',  100.0 - (100.0/base ) )
            pointer += 1

        self.__data["mfi_index"] = pd.Series(typ_price["mfi_index"].values, index = typ_price.index)


class Volatility(object):
    def __init__(self,symbol: str):
        try:
            path_norm_data ="./data/{}/normalized.csv".format(symbol)
            dataset = pd.read_csv(path_norm_data,index_col='index')
            
            self.__volatility = dataset['returns'].std() * math.sqrt(252)
            
        except:
            self.__volatility  = -1
            print("Unexpected error for symbol {}:{}".format(symbol, sys.exc_info()[0]))

    @property
    def annual(self):
        """ Get annualilzed volatility"""
        return self.__volatility

class SequenceBase(ABC):
    """
    Abstract Sequence base class.
    Reads csv file that has pre-processed normalized stock data and creates 
    and has abstract input (X) and target (Y) property methods

    Parameters
    ----------
    symbol: str
        Company trade symbol.  The symbol is also the root directory where
        the pre-processed normalized is located: ./data/{symbol}/quote_processed.csv
    """    
    
    def __init__(self,symbol:str, window_size:int, target_length:int):
        try:
            self.__window_size = window_size
            self.__target_length = target_length

            path_norm_data ="./data/{}/normalized.csv".format(symbol)
            self.__data_normal = pd.read_csv(path_norm_data,index_col='index')
        except:
            print("Unexpected error for symbol {}:{}".format(symbol, sys.exc_info()[0]))

    @property
    def data(self):
        """ Get the trade data as a DataFrame """ 
        return self.__data_normal

    @property
    def original_data(self):
        """ Get the original closing pricing as numpy arrays """ 
        return self.__data_normal['normal_close'].values

    @property
    def window_size(self):
        """ Get the feature sequence length""" 
        return self.__window_size

    @property
    def target_length(self):
        """ Get the target sequence length""" 
        return self.__target_length

    @property
    @abstractmethod
    def X(self):
        """ Get the input array """ 
        pass
     
    @property
    @abstractmethod
    def y(self):
        """ Get the target array """ 
        pass

class SimpleSequence(SequenceBase):
    """ Sequence class that creates input (x) and target (y) for RNN training or prediction,
        based on given window size (x) and target (y) lengths.
        The sequence is created from end of day normalized adjusted close stock pricess.

         Parameters
        ----------
        window_size: int
            The sequence length of the feature input (x).
        target_length: int
            The sequence length of the target variable (y).
    """
    def __init__(self,symbol:str, window_size:int, target_length:int):
        SequenceBase.__init__(self,symbol, window_size, target_length)
        self.__sequence_data()   
    
    def __sequence_data(self):
            """ Prepare normalized data into input (x) and target (y) sequence 
            """
            #normalized values
            close =  self.data['normal_close'].values

            X = []
            y = []

            pointer = 0
            data_length = len(close)
            while (pointer + self.window_size + self.target_length) <= data_length:
                
                #handle X inputs
                X.append(close[pointer:pointer + self.window_size])

                #handle target values
                y.append(close[pointer + self.window_size:pointer + self.window_size + self.target_length])
                
                pointer += 1

            self.__X = np.asarray(X)
            self.__X = self.__X.reshape((-1, self.__X.shape[-1], 1))
            self.__y = np.asarray(y)  #.reshape((-1,self.target_length))

    @property
    def X(self):
        """ Get the input array """ 
        return self.__X
     
    @property
    def y(self):
        """ Get the target array """ 
        return self.__y

class MultiSequence(SequenceBase):
    """ Sequence class that creates input (x) and target (y) for RNN training or prediction,
        based on given window size (x) and target (y) lengths.
        The sequence is created from three features i) end of day normalized adjusted close stock pricess
        ii) log normal returns and iii) normalized MFI index.

         Parameters
        ----------
        window_size: int
            The sequence length of the feature input (x).
        target_length: int
            The sequence length of the target variable (y). 
    """
    def __init__(self,symbol:str, window_size:int, target_length:int):
        SequenceBase.__init__(self,symbol, window_size, target_length)
        self.__sequence_data()   
    
    def __sequence_data(self):
        """ Prepare normalized data into input (x) and target (y) sequence 
        """
        #normalized log returns
        close = self.data['normal_close'].values #np.array(range(0,15))  #  
        returns = self.data['normal_returns'].values
        mfi = self.data['normal_mfi'].values

        X = []
        y = []

        pointer = 0
        data_length = len(close)
        while (pointer + self.window_size + self.target_length) <= data_length:
            
            #hadle X inputs
            x_close = close[pointer:pointer + self.window_size].reshape(-1,1)
            x_returns = returns[pointer:pointer + self.window_size].reshape(-1,1)
            x_mfi = mfi[pointer:pointer + self.window_size].reshape(-1,1)

            #combine feature sequences
            x_ = np.append(x_close,x_returns, axis=1)
            # x_ = np.append(x_close,x_mfi, axis=1)
            x_ = np.append(x_,x_mfi, axis=1)
            X.append(x_)

            #handle target values
            y.append(close[pointer + self.window_size:pointer + self.window_size + self.target_length])
            
            pointer += 1

        self.__X = np.asarray(X)
        self.__y = np.asarray(y)

    @property
    def X(self):
        """ Get the input array """ 
        return self.__X
     
    @property
    def y(self):
        """ Get the target array """ 
        return self.__y

#train 
def split_data(seq_obj: SequenceBase , split_rate = 0.2):

    """
    Splits the input and target sequence data between train and test
    base on split percentage    
    
    Parameter
    -----------
    seq_obj: Sequence object
            Has X (input) and y (target) propery method 
            corresponding to the input and target sequence data.
    split_rate: float
            The test split rate. Defualt of 0.20
    """
    split = int(len(seq_obj.X) * (1- split_rate))
    
    # partition the training set
    X_train = seq_obj.X[:split,:]
    y_train = seq_obj.y[:split]

    # keep the last chunk for testing
    X_test = seq_obj.X[split:,:]
    y_test = seq_obj.y[split:]
    
    return X_train,y_train,X_test,y_test

def graph_prediction(trained_model, X_train,X_test,original,window_size):
    # generate predictions for training
    train_predict = trained_model.predict(X_train)

    test_predict = trained_model.predict(X_test)

    # plot original series
    plt.plot(original, color='k')
    
    # plot training set prediction
    split = len(X_train)
    split_pt = split + window_size 
    train_in = np.arange(window_size,split_pt,1)
    plt.plot(train_in,train_predict,color = 'b')
    
    # plot testing set prediction
    test_in = np.arange(split_pt,split_pt + len(test_predict),1)
    plt.plot(test_in,test_predict,color = 'r')

    # pretty up graph
    plt.xlabel('day')
    plt.ylabel('(normalized) price of stock')
    plt.legend(['original series','training fit','testing fit'],loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()