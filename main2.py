#!/usr/bin/env python3

import csv

hourly_prices = []
r = csv.DictReader(open('Gemini_BTCUSD_1h.csv'))
for x in r:
  price = x['Close']
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
prevprice = first_price
#for price in hourly_prices[1:]:
#  total_squared_error
