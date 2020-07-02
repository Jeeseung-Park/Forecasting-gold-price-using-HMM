import os
import pandas as pd
import numpy as np


def get_data_path(symbol):
    # Return CSV file path given symbol.
    commodity_dir = 'data/commodities'
    currency_dir = 'data/currencies'

    if symbol in ['AUD', 'CNY', 'EUR', 'GBP', 'HKD', 'JPY', 'USD']:
        path = os.path.join(currency_dir, symbol + '.csv')
    else:
        path = os.path.join(commodity_dir, symbol + '.csv')

    return path


def merge_data(start_date, end_date, symbols):
    dates = pd.date_range(start_date, end_date)
    df = pd.DataFrame(index=dates)

    if 'Gold' not in symbols:
        symbols.insert(0, 'Gold')

    for symbol in symbols:
        df_temp = pd.read_csv(get_data_path(symbol), index_col="Date", parse_dates=True, na_values=['nan'])
        df_temp.columns = [symbol + '_' + col for col in df_temp.columns]  # rename columns
        df = df.join(df_temp)

    return df


def Make_Data(start_date, end_date, symbols_):
    table = merge_data(start_date, end_date, symbols=symbols_)  # 데이터셋 만들기 symbol 결정
    table = table[table['Gold_Price'].notnull()]  # index를 gold_price 기준으로
    table.drop([col for col in table.columns if col.split('_')[-1] in ['Volume', 'Change']], axis=1,
               inplace=True)  # Volume, Change, High, Low 삭제
    table.interpolate(inplace=True)  # USD NULL값 interpolation
    gold_price = table['Gold_Price']  # past_price용도

    return table, gold_price

def Make_RSI(df, symbols):
    for com in symbols:
        Price=df[com+'_Price']
        up= np.where(Price.diff(1) > 0, Price.diff(1), 0)
        down= np.where(Price.diff(1) < 0, Price.diff(1) *(-1), 0)
        Average_up=pd.DataFrame(up).rolling(14).mean()
        Average_down=pd.DataFrame(down).rolling(14).mean()
        RSI = (Average_up/(Average_down+Average_up)) *100
        RSI.index=df.index
        df[com+'_RSI']=RSI
    return 13 #Null 개수

def Make_ATR(df, symbols):
    for com in symbols:
        High=df[com+'_High']
        Low=df[com+'_Low']
        Close=df[com+'_Price']
        New_df=pd.DataFrame()
        New_df['tr0']=np.abs(High-Low)
        New_df['tr1']=np.abs(High-Close.shift())
        New_df['tr2']=np.abs(Low-Close.shift())
        True_range=New_df.max(axis=1)
        df[com+'_ATR']=True_range.ewm(alpha=1/14,adjust=False).mean()
    return 0 #Null 개수

def Make_BB(df,symbols):
    for com in symbols:
        Price=df[com+'_Price']
        MiddleBB=Price.rolling(20).mean()
        UpBB=MiddleBB+2*Price.rolling(20).std()
        LowBB=MiddleBB-2*Price.rolling(20).std()
        df[com+'_BB']=(UpBB-LowBB)/MiddleBB
    return 19 #Null 개수


def Make_ADX(df,symbols):
    for com in symbols:
        High=df[com+'_High']
        Low=df[com+'_Low']
        Close=df[com+'_Price']
        Up_move=High.diff(1)
        Down_move=-Low.diff(1)
        Plus_DM=np.where((Up_move>Down_move) & (Up_move>0), Up_move,0)
        Minus_DM=np.where((Up_move<Down_move)&(Down_move>0), Down_move,0)
        New_df=pd.DataFrame()
        New_df['tr0']=np.abs(High-Low)
        New_df['tr1']=np.abs(High-Close.shift())
        New_df['tr2']=np.abs(Low-Close.shift())
        True_range=New_df.max(axis=1)
        Plus_DM=pd.Series(Plus_DM,index=df.index)
        Minus_DM=pd.Series(Minus_DM,index=df.index)
        PDI=100*Plus_DM.ewm(14).mean()/True_range.ewm(alpha=1/14,adjust=False).mean()
        MDI=100*Minus_DM.ewm(14).mean()/True_range.ewm(alpha=1/14,adjust=False).mean()
        df[com+'_ADX']=100*np.abs(PDI-MDI).ewm(14).mean()/(PDI+MDI)
    return 1 #Null 개수는 1이지만 사실 13일 이전은 정확한 값은 아님.

def return_generator(column):
    diff = np.diff(column)
    price = np.array(column[:-1])
    return diff/price


def data_gold_price_generator(start_date, end_date, symbols):
    data, gold_price = Make_Data(start_date, end_date, symbols)

    null_max = max(Make_ADX(data, symbols), Make_ATR(data, symbols), Make_BB(data, symbols), Make_RSI(data, symbols))

    data = data[null_max:]
    data = data.reindex(sorted(data.columns), axis=1)
    data_index = data.index
    data = data.apply(lambda x: return_generator(x))
    data.index = data_index[1:]
    gold_price = gold_price[null_max + 1:]

    return data, gold_price


def training_set_generator(df, input_days, gold_price):
    length = df.shape[0]
    training_set = []

    for idx in range(length - input_days):  # except tail of the data.
        feature_set = np.array(df.iloc[idx: idx + input_days])
        observation = []
        for feature in zip(*feature_set):
            column = list(feature)[::-1]
            observation = observation + column
        observation = np.array(observation)
        training_set.append(observation)

    refined_gold_price = gold_price[input_days - 1:]

    return training_set, refined_gold_price

input_days = 6
symbols = ['Gold']


def make_features(start_date, end_date, is_training):
    data, gold_price = data_gold_price_generator(start_date, end_date, symbols)
    training_sets, refined_gold_price = training_set_generator(data, input_days, gold_price)

    training_x = training_sets[:-10]

    test_x = training_sets[-10:]
    past_price = refined_gold_price[-11:-1]
    target_price = refined_gold_price[-10:]

    return training_x if is_training else (test_x, past_price, target_price)


if __name__ == "__main__":
    start_date = '2010-01-01'
    end_date = '2020-04-18'

    make_features(start_date, end_date, is_training=False)