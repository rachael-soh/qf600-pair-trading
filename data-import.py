import wrds 
import psycopg2

db = wrds.Connection(wrds_username='sohrac')

# data = db.raw_sql("select * from crsp.dsenames limit 100")
data = db.raw_sql("""
    select a.permno, a.date, a.ret, a.prc, a.vol,
       b.shrcd, b.exchcd, b.siccd,
       c.dlret
from crsp.dsf a
left join crsp.dsenames b
  on a.permno = b.permno
 and a.date >= b.namedt
 and a.date <= b.nameendt
left join crsp.msedelist c
  on a.permno = c.permno
 and a.date = c.dlstdt
where a.date between '2020-01-01' and '2021-12-31'
  and b.shrcd in (10,11)
  and b.exchcd in (1,2,3)
""", date_cols=['date'])
data.to_csv('crsp_joined_data.csv', index=False)
