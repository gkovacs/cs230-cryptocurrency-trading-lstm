# cryptocurrency-trading

## About

This repository contains our final project for cs230. It is a deep neural network (LSTM network implemented in Keras) that predicts the future price of Bitcoin (in USD) based on historic order books data on the Gemini cryptocurrency exchange.

After cloning this repository, you will need to get the gemini order book dataset. The dataset we used are order books from 10/08/2015 - 02/20/2018. You can buy this dataset at https://datashop.cboe.com/cryptocurrency-gemini-order-book-data

More details can be found in [poster.jpg](https://github.com/gkovacs/cryptocurrency-trading/blob/master/poster.jpg) and [report.pdf](https://github.com/gkovacs/cryptocurrency-trading/blob/master/report.pdf)

## Preprocessing

The gemini orders book dataset comes as a bunch of zip files that you download from their FTP server

You first want to convert those into parquet files - use the scripts [`convert_to_lz4.py`](https://github.com/gkovacs/cryptocurrency-trading/blob/master/convert_to_lz4.py.py) and [`to_parquet.py`](https://github.com/gkovacs/cryptocurrency-trading/blob/master/to_parquet.py) to do this

Then you can run [`preprocessing.ipynb`](https://github.com/gkovacs/cryptocurrency-trading/blob/master/preprocessing.ipynb) to extract out a bunch of features from the order books

## Models

The training and testing procedure for the best model we found is in [`training_testing_FINAL_MODEL.ipynb`](https://github.com/gkovacs/cryptocurrency-trading/blob/master/training_testing_FINAL.ipynb)

The [models](https://github.com/gkovacs/cryptocurrency-trading/tree/master/models) directory contains several pre-trained models that can be loaded via Keras

## License

MIT

## Contact

[Geza Kovacs](http://github.com/gkovacs)


