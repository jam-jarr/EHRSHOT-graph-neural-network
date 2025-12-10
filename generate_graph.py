import sqlite3
from collections import defaultdict

import pandas as pd
import torch
from torch_geometric.data import Data

df = pd.read_csv("./query.csv")

patients_to_meds = defaultdict(list)
unique_medications = set()
unique_patients = set()

# Generate adjacency list
for i in range(len(df)):
    row = df.iloc[i]
    patient = int(row["person_id"])
    medication = int(row["drug_component_id"])
    patients_to_meds[patient].append(medication)

    # node, node_label
    unique_patients.add((patient, patient))
    med_name = row["drug_name"]
    unique_medications.add((medication, med_name))

# patients_to_meds = {
#   patient1 = { medicationA: 1, medicationB: 2, ... },
#   ...
# }

print(patients_to_meds)

edge_list = [[], []]
# node_features = []

# Generate edge_list
for p, meds in patients_to_meds.items():
    for m in meds:
        edge_list[0].append(p)
        edge_list[1].append(m)


edge_index = torch.tensor(edge_list, dtype=torch.long)
# edge_attr = torch.tensor(edge_weights, dtype=torch.int64)


def pyg_data_to_edges_csv(data, filename="edges.csv"):
    if data.edge_index is None or data.edge_index.numel() == 0:
        print("No edges to export")
        return

    src, dst = data.edge_index
    src = src.cpu().numpy()
    dst = dst.cpu().numpy()

    edge_data = {"source": src, "target": dst}

    # Add edge attributes (e.g., weights) if present
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        edge_attr = data.edge_attr.cpu().numpy()
        for i in range(edge_attr.shape[1]):
            edge_data["weight"] = edge_attr[:, i]

    df = pd.DataFrame(edge_data)
    df.to_csv(filename, index=False)
    print(f"Saved {len(src)} edges to {filename}")


nodes = []
node_label = []
node_type = []

# for i in range(0, len(medicationdf)):
#     med = medicationdf.iloc[i]
#     drug_id = med["drug_component_id"]
#     nodes.append(drug_id)
#     drug_name = med["drug_name"]
#     node_label.append(drug_name)

# Append medications to node list
for medid, med_name in unique_medications:
    nodes.append(medid)
    node_label.append(med_name)
    node_type.append("medication")

# Append patients to node list
# TODO: implement proper identifier
for patientid, identifier in unique_patients:
    nodes.append(patientid)
    node_label.append(identifier)
    node_type.append("patient")


node_data = {"id": nodes, "label": node_label, "type": node_type}

df = pd.DataFrame(node_data)

df.to_csv("nodes.csv", index=False)

data = Data(edge_index=edge_index)

pyg_data_to_edges_csv(data)
