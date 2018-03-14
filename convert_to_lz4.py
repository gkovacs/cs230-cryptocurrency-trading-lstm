#!/usr/bin/env python3

from glob import glob
from plumbum.cmd import gunzip, lz4, rm

for x in glob('cboe/lz4/*.csv.gz'):
  print(x)
  gunzip[x]()
  csvfile = x.replace('.csv.gz', '.csv')
  outfile = x.replace('.csv.gz', '.csv.lz4')
  (lz4['-9', csvfile] > outfile)()
  rm(csvfile)
