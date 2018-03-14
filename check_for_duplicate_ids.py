#!/usr/bin/env python3

import glob
import gzip
import io
import lz4.frame
import csv
import numpy as np

#import diskcache as dc
#cache = dc.Cache('tmp')

trading_pairs = ['BTCUSD', 'ETHUSD', 'ETHBTC']
for trading_pair in trading_pairs:
  fz = io.TextIOWrapper(lz4.frame.open('cboe/' + trading_pair + '_duplicates_removed.csv.lz4', mode='wb'), encoding='utf-8')
  #fz = open('cboe/' + trading_pair + '_duplicates_removed.csv', 'wt')
  allfiles = glob.glob('cboe/lz4/' + trading_pair + '*.csv.lz4')
  if len(allfiles) == 0:
    continue
  fieldnames = None
  for filename in allfiles:
    with io.TextIOWrapper(lz4.frame.open(filename, 'rb'), encoding='utf-8') as file:
      reader = csv.reader(file)
      for x in reader:
        print(x)
        fieldnames = x
        break
      break

  maxid = 0
  for filename in allfiles:
    print(filename)
    with io.TextIOWrapper(lz4.frame.open(filename, 'rb'), encoding='utf-8') as file:
      reader = csv.DictReader(file)
      for x in reader:
        id = int(x['Event ID'])
        maxid = max(id, maxid)
  seen_ids = np.full(maxid + 1, False, dtype=bool)

  writer = csv.DictWriter(fz, fieldnames)
  writer.writeheader()

  #ids = set()
  allfiles = glob.glob('cboe/lz4/' + trading_pair + '*.csv.lz4')
  for filename in allfiles:
    print(filename)
    with io.TextIOWrapper(lz4.frame.open(filename, 'rb'), encoding='utf-8') as file:
      reader = csv.DictReader(file)
      for x in reader:
        id = int(x['Event ID'])
        if seen_ids[id]:
          continue
        seen_ids[id] = True
        #if id in cache:
        #	continue
        #cache[id] = True
        writer.writerow(x)
