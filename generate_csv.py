import sqlite3

import pandas as pd

conn = sqlite3.connect(
    "../EHR2000.db"
)  # Connect (and create if it doesn't exist) database
curr = conn.cursor()  # Object to run queries

full_drug_group = "project_full_sampled_data_dose_group"

table = full_drug_group

df = pd.read_sql_query(
    f"""
    SELECT person_id, drug_component_id, drug_name, COUNT(drug_component_id), gender_concept_id, year_of_birth, race_concept_id FROM {table}
    WHERE drug_component_id != 0
    GROUP BY drug_component_id, person_id
    ORDER BY person_id
    """,
    conn,
)


conn.close()

df.to_csv("query.csv", index=False)
