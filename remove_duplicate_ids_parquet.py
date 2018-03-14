#!/usr/bin/env python3

import paratext
import pandas
import lz4.frame
import gzip
import io
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

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

trading_pairs = ['BTCUSD', 'ETHUSD', 'ETHBTC']

for trading_pair in trading_pairs:
  allfiles = sorted(glob(f'cboe/parquet/{trading_pair}*.parquet'))

  print(f'trading pair {trading_pair} finding max id')
  maxid = 0

  for x in allfiles:
    table = pq.read_table(x, columns=['Event ID']).to_pandas()
    curmax = table['Event ID'].max()
    maxid = max(maxid, curmax)

  print(f'max id for {trading_pair} is {maxid}')
  seen_ids = np.full(maxid + 1, False, dtype=bool)

  for x in allfiles:
    outfile = x.replace('cboe/parquet/', 'cboe/parquet_nodups/')
    print(outfile)
    table = pq.read_table(x).to_pandas()
    def is_duplicate(row):
      id = row['Event ID']
      retval = seen_ids[id]
      if not retval:
        seen_ids[id] = True
      return retval
    table['isduplicate'] = table.apply(is_duplicate, axis=1)
    table = table.query('isduplicate == False')
    del table['isduplicate']
    pq.write_table(pa.Table.from_pandas(table), outfile, compression='snappy')
