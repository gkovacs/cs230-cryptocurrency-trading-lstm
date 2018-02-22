#!/usr/bin/env python3

import csv

hourly_prices = []
r = csv.DictReader(open('Gemini_BTCUSD_1h.csv'))
for x in r:
  price = float(x['Close'])
  hourly_prices.append(price)
hourly_prices.reverse()
first_price = None
last_price = None
for price in hourly_prices:
  if first_price == None:
    first_price = price
  last_price = price
print('first price')
print(first_price)
print('last price')
print(last_price)

upevents = 0
downevents = 0
sameprice = 0
prevprice = first_price
for price in hourly_prices[1:]:
  if price > prevprice:
    upevents += 1
  if price < prevprice:
    downevents += 1
  if price == prevprice:
    sameprice += 1
  prevprice = price
print('upevents')
print(upevents)
print('downevents')
print(downevents)
print('sameprice')
print(sameprice)

total_squared_error = 0
total_absolute_error = 0
prevprice = first_price
for price in hourly_prices[1:]:
  total_squared_error += (price - prevprice)**2
  total_absolute_error += abs(price - prevprice)
  prevprice = price
num_predictions = len(hourly_prices) - 1
mean_squared_error = total_squared_error / num_predictions
mean_absolute_error = total_absolute_error / num_predictions
print('=== baseline on everything ===')
print('total squared error')
print(total_squared_error)
print('total absolute error')
print(total_absolute_error)
print('mean squared error')
print(mean_squared_error)
print('mean absolute error')
print(mean_absolute_error)

traindata = hourly_prices[:int(len(hourly_prices)*8/10)]
devdata = hourly_prices[int(len(hourly_prices)*8/10):int(len(hourly_prices)*9/10)]
testdata = hourly_prices[int(len(hourly_prices)*9/10):]

prevprice = devdata[0]
total_squared_error = 0
total_absolute_error = 0
for price in devdata[1:]:
  total_squared_error += (price - prevprice)**2
  total_absolute_error += abs(price - prevprice)
  prevprice = price
num_predictions = len(hourly_prices) - 1
mean_squared_error = total_squared_error / num_predictions
mean_absolute_error = total_absolute_error / num_predictions
print('=== baseline on dev set ===')
print('total squared error')
print(total_squared_error)
print('total absolute error')
print(total_absolute_error)
print('mean squared error')
print(mean_squared_error)
print('mean absolute error')
print(mean_absolute_error)

prevprice = testdata[0]
total_squared_error = 0
total_absolute_error = 0
for price in testdata[1:]:
  total_squared_error += (price - prevprice)**2
  total_absolute_error += abs(price - prevprice)
  prevprice = price
num_predictions = len(hourly_prices) - 1
mean_squared_error = total_squared_error / num_predictions
mean_absolute_error = total_absolute_error / num_predictions
print('=== baseline on test set ===')
print('total squared error')
print(total_squared_error)
print('total absolute error')
print(total_absolute_error)
print('mean squared error')
print(mean_squared_error)
print('mean absolute error')
print(mean_absolute_error)
