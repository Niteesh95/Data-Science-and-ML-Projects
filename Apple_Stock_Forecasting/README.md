# Apple_Stock_Prediction

Apple stock prediction using ARIMA and LSTM models. Data is obtained using pandas data reader and yahoo finance.
1. EDA on stock data.
2. Autoregressor, Moving Average, Autocorrelation, ACF, PACF plots.
3. Deep learning LSTM model to predict Apple's closing stock price.

Libraries: pandas, NumPy, matplotlib, seaborn, keras, tensorflow, ARIMA, SARIMAX, acf, pacf

## EDA
Simple moving average - let us consider a window size = 10, the SMA is given by taking average of the first 10 records and putting the result in the 11th record. next for the 12th record, the window is shifted on space below, i.e the average of 2 -11th records is taken.
<p align='center'>
  <img src='https://user-images.githubusercontent.com/60603790/214404919-fdede582-8b8a-481e-81d4-7ebca0797336.png' width='700' height='300' />
</p>

As we can see after differencing by 1 order, the data seems to be somewhat stationary.
<p align='center'>
  <img src='https://user-images.githubusercontent.com/60603790/214418491-99e3331e-0b3b-46cf-abc2-c859d6b8deb2.png' width='700' height='300' />
</p>

## Modelling and Forecasting
The LSTM model forecasting on validation set and the grouth truth are shown in the below graph.
<p align='center'>
  <img src='https://user-images.githubusercontent.com/60603790/214418774-2b95fb40-a35f-475f-bfb8-09c27f3f45b0.png' width='700' height='300' />
</p>


