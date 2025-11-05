import wrds
import pandas as pd
def fetch_crsp_data(db: wrds.Connection, start_date: str, end_date: str) -> pd.DataFrame:
    sql = f""" SELECT 
        a.permno, a.permco, a.date, a.prc, a.ret, a.vol, a.shrout, a.cfacpr,
        a.cfacshr, b.siccd, b.shrcd, b.exchcd, c.dlret,
        ABS(a.prc)/a.cfacpr as adj_prc
    FROM crsp.dsf a
    LEFT JOIN crsp.dsenames b
        ON a.permno = b.permno
        AND a.date BETWEEN b.namedt AND b.nameendt
    LEFT JOIN crsp.dsedelist c
        ON a.permno = c.permno
        AND a.date = c.dlstdt
    WHERE 
        a.date BETWEEN '{start_date}' AND '{end_date}'
        AND b.shrcd IN (10, 11) 
        AND b.exchcd IN (1, 2, 3)
    """
    df = db.raw_sql(sql, date_cols=['date'])
    return df

if __name__ == "__main__":
    db = wrds.Connection(wrds_username='sohrac')
    # TODO: fetch dates needed
    crsp_data = fetch_crsp_data(db, '1961-01-01', '1973-07-31')
    crsp_data.to_csv("crsp_daily_1961_1973.csv")
