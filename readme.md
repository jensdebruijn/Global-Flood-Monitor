## Abstract
Early event detection and response can significantly reduce the societal impact of floods. Currently, early warning systems rely on gauges, radar data, models and informal local sources. However, the scope and reliability of these systems are limited. Recently, the use of social media for detecting disasters has shown promising results, especially for earthquakes. Here, we present a new database for detecting floods in real-time on a global scale using Twitter. The method was developed using 88 million tweets, from which we derived over 10.000 flood events (i.e., flooding occurring in a country or first order administrative subdivision) across 176 countries in 11 languages in just over four years. Using strict parameters, validation shows that approximately 90% of the events were correctly detected. In countries where the first official language is included, our algorithm detected 63% of events in NatCatSERVICE disaster database at admin 1 level. Moreover, a large number of flood events not included in NatCatSERVICE are detected. All results are publicly available on www.globalfloodmonitor.org.

## Cite as
Bruijn, J.A., Moel, H., Jongman, B. et al. A global database of historic and real-time flood events based on social media. Sci Data 6, 311 (2019) doi:10.1038/s41597-019-0326-9

## Links
 - [Flood Tweet IDs (multilingual)](https://doi.org/10.7910/DVN/T3ZFMR)
 - [Historic flood event database](https://doi.org/10.5281/zenodo.3525033)
 - [Real-time flood event database](https://www.globalfloodmonitor.org)

## How to run
1. Setup
    - Install Python (3.6+) and all modules in `requirements.txt`.
    - Install PostgreSQL (tested with 12) and PostGIS (tested with 3.0).
    - Set all parameters in `config.py`. This includes the `TWITTER_CONSUMER_KEY`, `TWITTER_CONSUMER_SECRET`, `TWITTER_ACCESS_TOKEN`, `TWITTER_ACCESS_TOKEN_SECRET` for which you will need to register as [Twitter developer](https://developer.twitter.com/).
2. Preparing data & preprocessing
    - Obtain shapefiles for countries (`input/regions/level0.shp`) and first order administrative subdivisions (`input/regions/level0.shp`). The column 'ID' should be the geonames ID prefixed with 'g-' (e.g., `g-2750405` for the Netherlands).
    - Set all parameters in `config.py`
    - Create elasticsearch index for tweets using create_index.py. This file automatically uses the proper index settings (see `input/es_document_index_settings.json`).
    - Fill index with tweets (example for reading tweets from jsonlines to database in `fill_es.py`). This assumes the file `input/example.jsonl` has a new json-object obtained from the Twitter API on each line.
    - Run `preprocessing.py`
3. Creating the text classifier
    - Hydrate the labelled data (*input/labeled_tweets.xlsx*) by running `hydrate.py`. This creates a new file with additional data obtained from the Twitter API (including the tweets' texts in `input/labeld_tweets_hydrated.xlsx`). Don't forget to set the Twitter developer tokens in `config.py`
    - Train the classifier by running `train_text_classifier.py`. This file exports the trained classifier to *input/classifier*.
4. Finding time corrections per region
    - In the next step we need to run just the localization algorithm [TAGGS](https://github.com/jensdebruijn/TAGGS) so that we can derive the number of localized tweets per hour of the day (see paper). To do so we run the main file `main.py`, with detection set to false, like so: `main.py --detection false`
    - Run `get_time_correction.py`. This will create a new file `input/time_correction.json`.
5. Run the Global Flood Monitor
    - Finally, run `main.py` without arguments to run the Global Flood Monitor. The resulting events are stored in the PostgreSQL database.

## Contact
Jens de Bruijn -- j.a.debruijn at outlook dot com