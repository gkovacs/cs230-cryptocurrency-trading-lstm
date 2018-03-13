#!/usr/bin/env python3

import glob
import gzip
import csv

allfiles = glob.glob('cboe/gzipped/*.csv.gz')
for filename in allfiles:
  with gzip.open(filename, 'rt') as file:
    reader = csv.DictReader(file)
    for x in reader:
      print(x)

