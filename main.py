#!/usr/bin/env python3

import csv

upevents = 0
downevents = 0
prevprice = 0
r = csv.DictReader(open('BTCUSD_order_book_20170210.csv'))
for x in r:
  t = x['Event Type']
  if t == 'Fill':
    price = x['Avg Price (USD)']
    print(price)
    