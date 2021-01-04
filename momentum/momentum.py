# import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

# Import the backtrader platform
import backtrader as bt


# Create a Stratey
class TestStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
    
    def calculate_momentum(self):
        '''
        Calculate momentum (stock price change in %) for 3, 6 and 12 months.
        Return a tuple of three lists containing:
            - the calculated percentages (list of floats)
            - whether momentums are positive or not (list of bools)
            - whether momentums are increasing per time period (list of bools)
        '''
        
        # Decide if momentum should be based on current price or last month's
        ref_period = 0  # 0 = current, -1 = last month
        # ref_period = -1  # 0 = current, -1 = last month
        
        # Calculate momentum based on 3, 6 and 12 months
        self.mom_3 = (self.dataclose[ref_period] - self.dataclose[-3]) / self.dataclose[-3]
        self.mom_6 = (self.dataclose[ref_period] - self.dataclose[-6]) / self.dataclose[-6]
        self.mom_12 = (self.dataclose[ref_period] - self.dataclose[-12]) / self.dataclose[-12]
        
        # Calculate momentum for stock
        return ([self.mom_3, self.mom_6, self.mom_12], 
                [self.mom_3 > 0, self.mom_6 > 0, self.mom_12 > 0],
                [self.mom_6 >= self.mom_3, self.mom_12 >= self.mom_6])
    
    def next(self):
        
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])
        
        # Collect 12 months of data before trading
        # Act only every 3:rd month
        if len(self) >= 12 and len(self) % 3 == 0:
                
                # First, sell eveything
                if self.position:
                    self.log(f'SELL CREATE, {self.dataclose[0]} (> {self.dataclose[-3]} (3m) > {self.dataclose[-6]} (6m) > {self.dataclose[-12]} (12m))')
                    self.sell()
                    
                    # Calculate momentum
                    # ([m3, m6, m12],
                    # [m3_pos, m6_pos, m12_pos],
                    # [m6_gt_m3, m12_gt_m6]) = self.calculate_momentum
                
                # Only buy if positive momentum (12m > 6m > 3m)
                if self.dataclose[0] > self.dataclose[-3] > self.dataclose[-6] > self.dataclose[-12]:
                    
                    self.log(f'BUY CREATE, {self.dataclose[0]} (> {self.dataclose[-3]} (3m) > {self.dataclose[-6]} (6m) > {self.dataclose[-12]} (12m))')
                    self.buy()
                else:
                    self.log('NO MOMENTUM IN STOCK :(')


if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(TestStrategy)

    # Datas are in a subfolder of the samples. Need to find where the script is
    # because it could have been called from anywhere
    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    datapath = os.path.join(modpath, 'datas/orcl-1995-2014.txt')

    # Create a Data Feed
    data = bt.feeds.YahooFinanceCSVData(
        dataname=datapath)

    # # Create a Data Feed
    # data = bt.feeds.YahooFinanceCSVData(
    #     dataname=datapath,
    #     # Do not pass values before this date
    #     fromdate=datetime.datetime(1995, 1, 1),
    #     # Do not pass values before this date
    #     todate=datetime.datetime(1995, 12, 31),
    #     # Do not pass values after this date
    #     reverse=False)

    # Resample
    cerebro.resampledata(
        data,
        timeframe=bt.TimeFrame.Months)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())