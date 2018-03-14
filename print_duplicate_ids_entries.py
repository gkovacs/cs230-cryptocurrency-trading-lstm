#!/usr/bin/env python3

import paratext
import pandas
import lz4.frame
import gzip
import io
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import copy

'''
filepath = 'cboe/lz4_test/BTCUSD_order_book_20170627.csv.lz4'
#filepath = 'cboe/lz4_test/BTCUSD_order_book_20170627.csv.gz'
df = pandas.read_csv(io.TextIOWrapper(lz4.frame.open(filepath)))
#df = pandas.read_csv(filepath)
#df = paratext.load_csv_to_pandas(gzip.open(filepath).read())
print((df))
'''

from glob import glob
from plumbum.cmd import rm
import sys

trading_pairs = ['BTCUSD', 'ETHUSD', 'ETHBTC']

for trading_pair in trading_pairs:
  allfiles = sorted(glob(f'cboe/parquet/{trading_pair}*.parquet'))

  id_to_row = {}
  id_to_filesrc = {}

  for x in allfiles:
    outfile = x.replace('cboe/parquet/', 'cboe/parquet_nodups/')
    print(outfile)
    table = pq.read_table(x).to_pandas()
    def is_duplicate(row):
      id = row['Event ID']
      #if id == 343:
      #  print(row)
      retval = id in id_to_row
      if retval:
        print(x)
        print(row)
        print(id_to_filesrc[id])
        print(id_to_row[id])
        sys.exit()
      else:
        id_to_row[id] = copy.copy(row)
        id_to_filesrc[id] = copy.copy(x)
      return retval
    table['isduplicate'] = table.apply(is_duplicate, axis=1)
    table = table.query('isduplicate == False')
    del table['isduplicate']
