#!/usr/bin/env python3

import paratext
import pandas as pd
import lz4.frame
import gzip
import io
import pyarrow.parquet as pq
import pyarrow as pa

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

all_dataframes = []
for x in sorted(glob('cboe/parquet_fills_only_BTCUSD/*.parquet')):
  print(x)
  df = pq.read_table(x).to_pandas()
  all_dataframes.append(df)
result = pd.concat(all_dataframes)
pq.write_table(pa.Table.from_pandas(result), 'cboe/parquet_fills_only_BTCUSD.parquet', compression='snappy')

'''
for x in sorted(glob('cboe/parquet_fills_only_BTCUSD/*.parquet')):
  print(x)
  df = pq.read_table(x).to_pandas()
  print(df.dtypes.to_dict())
'''
