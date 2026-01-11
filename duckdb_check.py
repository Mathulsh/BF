import duckdb
import pandas as pd
from pandas import DataFrame

conn: duckdb.DuckDBPyConnection = duckdb.connect("/Users/lishihong/projects/Research/HEA/acln/src/acln/results1.duckdb")
count = conn.execute("SELECT COUNT(*) FROM results").fetchall()[0][0]
print(f"Number of records in results table: {count}")

top10: DataFrame = conn.sql("SELECT * FROM results ORDER BY mean_f1_macro DESC LIMIT 10;").fetchdf()
print("Top 10 records:", top10)

# df = conn.sql("SELECT * FROM results LIMIT 10").fetchdf()
# print(df)
# conn.execute("CREATE TABLE IF NOT EXISTS result AS SELECT * FROM df;");
# print(conn.sql("SELECT * FROM results"))

# C(43,4) = 124100,实际123409，缺691条