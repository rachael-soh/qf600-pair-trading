import wrds 
import psycopg2
import pandas as pd
from pairs_formation import *

db = wrds.Connection(wrds_username='sohrac')

# data = db.raw_sql("select * from crsp.dsenames limit 100")
# dsf: Daily Stock - Securities
# dsenames: CRSP Daily Stock Event - Name History, 
# dsedelist: CRSP Daily Stock Event - Delisting	
# priamry key: permco 
# data = db.raw_sql("""
#     select a.permco, a.date, a.ret, a.prc, a.vol,
#        b.comnam, b.shrcd, b.exchcd, b.siccd,
#        c.dlret
#     from crsp.dsf a
#       left join crsp.dsenames b
#         on a.permco = b.permco
#       and a.date >= b.namedt
#       and a.date <= b.nameendt
#       left join crsp.dsedelist c
#         on a.permco = c.permco
#       and a.date = c.dlstdt
#       where a.date between '2020-01-01' and '2021-12-31'
#         and b.shrcd in (10,11)
#         and b.exchcd in (1,2,3)
# """, date_cols=['date'])
# data.to_csv('crsp_joined_data.csv', index=False)
def get_stock(df) -> pd.DataFrame:
  permcos = set(df['permco_1'].unique()).union(df['permco_2'].unique())
  # comnames = db.raw_sql(
  #   f""" 
  #   SELECT DISTINCT ON (permco) *
  #   FROM crsp.dsenames
  #   WHERE permco IN ({','.join(map(str, permcos))})
  #   ORDER BY permco, namedt DESC
  #   """
  # )
  # df = df.merge(comnames.rename(columns=lambda x: x+"_1"), how='left', left_on='permco_1', right_on='permco_1')
  # df = df.merge(comnames.rename(columns=lambda x: x+"_2"), how='left', left_on='permco_2', right_on='permco_2')
  comnames = db.raw_sql(
    f""" 
    SELECT distinct on (permco) comnam, permco, ticker
    FROM crsp.dsenames
    WHERE permco IN ({','.join(map(str, permcos))})
    ORDER BY permco, namedt DESC
    """
  )
  df = df.merge(comnames, how='left', left_on='permco_1', right_on='permco').rename(columns={"comnam": "comnam_1","ticker": "ticker_1"})
  df = df.merge(comnames, how='left', left_on='permco_2', right_on='permco').rename(columns={"comnam": "comnam_2","ticker": "ticker_2"})
  df = df.drop(columns=['permco_x', 'permco_y'])
  return df
    

matched_pairs = pd.read_csv("matched_pairs.csv")
matched_pairs_named = get_stock(matched_pairs)
matched_pairs_named.to_csv("matched_pairs_with_name_ticker.csv", index=False)