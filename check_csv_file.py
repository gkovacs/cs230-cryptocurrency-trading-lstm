#!/usr/bin/env python3

import csv

r = csv.DictReader(open('BTCUSD_order_book_20171021.csv'))
for x in r:
  field = 'Limit Price (USD)'
  try:
    print(float(x[field]))
  except:
    print(x)
    break

