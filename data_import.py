import wrds 
import psycopg2
import pandas as pd
# from pairs_formation import *


def get_stock(df) -> pd.DataFrame:
  db = wrds.Connection(wrds_username='sohrac')
  permcos = set(df['permco_1'].unique()).union(df['permco_2'].unique())
  comnames = db.raw_sql(
    f""" 
    SELECT distinct on (permco) comnam, permco, ticker, naics
    FROM crsp.dsenames
    WHERE permco IN ({','.join(map(str, permcos))})
    ORDER BY permco, namedt DESC
    """
  )
  df = df.merge(comnames, how='left', left_on='permco_1', right_on='permco').rename(columns={"comnam": "comnam_1","ticker": "ticker_1","naics": "naics_1"})
  df = df.merge(comnames, how='left', left_on='permco_2', right_on='permco').rename(columns={"comnam": "comnam_2","ticker": "ticker_2","naics": "naics_2"})
  df = df.drop(columns=['permco_x', 'permco_y'])
  return df


if __name__ == "__main__":
  matched_pairs = pd.read_csv("data\matched_pairs_2024-01-01_2024-12-31.csv")
  matched_pairs_named = get_stock(matched_pairs)
  matched_pairs_named.to_csv("data\matched_pairs_with_name_ticker_2024-01-01_2024-12-31.csv", index=False)