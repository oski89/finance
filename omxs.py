import bs4 as bs
from collections import Counter
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests
# from sklearn import svm, model_selection as cross_validation, neighbors
# from sklearn.ensemble import VotingClassifier, RandomForestClassifier

### Something is wrong with TIGO SDB ###

style.use('ggplot')


def save_omxs_large_cap_tickers():
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:61.0) Gecko/20100101 Firefox/61.0'}
    resp = requests.get('http://www.nasdaqomxnordic.com/shares/listed-companies/nordic-large-cap', headers=headers)
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'id': 'listedCompanies'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[1].text
        currency = row.findAll('td')[2].text
        if currency == 'SEK' and ticker != 'TIGO SDB':
            tickers.append(ticker)

    with open('omxs-large-cap-tickers.pickle', 'wb') as f:
        pickle.dump(tickers, f)

    return tickers


def get_data_from_yahoo(reload_omxs_large_cap=False):
    if reload_omxs_large_cap:
        tickers = save_omxs_large_cap_tickers()
    else:
        with open('omxs-large-cap-tickers.pickle', 'rb') as f:
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2018, 12, 31)

    for ticker in tickers:
        yahoo_ticker = ticker.replace(' ', '-') + '.ST'

        if not os.path.exists(f'stock_dfs/{ticker}.csv'):
            df = web.DataReader(yahoo_ticker, 'yahoo', start, end)
            df.to_csv(f'stock_dfs/{ticker}.csv')
            print(f'Fetched {ticker}.')
        else:
            print(f'Already have {ticker}.')


def compile_data():
    with open('omxs-large-cap-tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv(f'stock_dfs/{ticker}.csv')
        df.set_index('Date', inplace=True)
        df.rename(columns={'Adj Close': ticker}, inplace=True)
        df.drop(['High', 'Low', 'Open', 'Close', 'Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

    main_df.to_csv('omxs-large-cap-joined-closes.csv')


def visualize_data():
    df = pd.read_csv('omxs-large-cap-joined-closes.csv')
    df_corr = df.corr()

    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1, 1)
    plt.tight_layout()
    plt.show()


def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('omxs-large-cap-joined-closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days + 1):
        df[f'{ticker}_{i}d'] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df, hm_days


def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > 0.029:
            return 1
        if col < -0.027:
            return -1
    return 0


def extract_featuresets(ticker):
    tickers, df, hm_days = process_data_for_labels(ticker)

    df[f'{ticker}_target'] = list(map(buy_sell_hold, *[df[f'{ticker}_{i}d'] for i in range(1, hm_days + 1)]))

    vals = df[f'{ticker}_target'].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))
    df.fillna(0, inplace=True)

    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df[f'{ticker}_target'].values

    return X, y, df


def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)

    # clf = neighbors.KNeighborsClassifier()

    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('Accuracy', confidence)

    predictions = clf.predict(X_test)
    print('Predicted spread:', Counter(predictions))

    return confidence


save_omxs_large_cap_tickers()
# get_data_from_yahoo(True)
# compile_data()
# visualize_data()
# process_data_for_labels('ABB')
# extract_featuresets('ABB')
# do_ml('ABB')
