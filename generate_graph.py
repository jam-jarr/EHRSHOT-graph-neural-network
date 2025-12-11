from collections import defaultdict

import pandas as pd
import torch
from torch_geometric.data import Data

df = pd.read_csv("./query.csv")

patients_to_meds = defaultdict(list)
unique_medications = set()
unique_patients = set()

ablation = "onlyage"

# Generate adjacency list
for i in range(len(df)):
    row = df.iloc[i]
    patient = int(row["person_id"])
    medication = int(row["drug_component_id"])
    patients_to_meds[patient].append(medication)

    gender = row["gender_concept_id"]
    age = row["year_of_birth"]
    race = row["race_concept_id"]

    concatfeature = f"{gender}_{age}_{race}"

    match ablation:
        case "concatfeature":
            unique_patients.add((patient, concatfeature))
        case "fullfeature":
            unique_patients.add((patient, patient, gender, age, race))
        case "nofeature":
            unique_patients.add((patient, patient))
        case "onlyage":
            unique_patients.add((patient, age))
    med_name = row["drug_name"]
    unique_medications.add((medication, med_name))

# patients_to_meds = {
#   patient1 = { medicationA: 1, medicationB: 2, ... },
#   ...
# }


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
node_age = []
node_gender = []
node_race = []


# Append medications to node list
for medid, med_name in unique_medications:
    nodes.append(medid)
    node_label.append(med_name)
    node_type.append("medication")

# Append patients to node list
for p in unique_patients:
    patientid = p[0]
    nodes.append(patientid)
    node_type.append("patient")

    match ablation:
        case "concatfeature":
            node_label.append(p[1])
        case "fullfeature":
            node_age = p[1]
            node_gender = p[2]
            node_race = p[3]
        case "nofeature":
            node_label.append(p[1])
        case "onlyage":
            node_label.append(p[1])

match ablation:
    case "concatfeature":
        node_data = {"id": nodes, "label": node_label, "type": node_type}
    case "fullfeature":
        node_data = {
            "id": nodes,
            "type": node_type,
            "age": node_age,
            "gender": node_gender,
            "race": node_race,
        }
    case "nofeature":
        node_data = {"id": nodes, "label": node_label, "type": node_type}
    case "onlyage":
        node_data = {"id": nodes, "label": node_label, "type": node_type}


if ablation != "fullfeature":
    node_data = {"id": nodes, "label": node_label, "type": node_type}
else:
    node_data = {
        "id": nodes,
        "type": node_type,
        "age": node_age,
        "gender": node_gender,
        "race": node_race,
    }


df = pd.DataFrame(node_data)

df.to_csv("nodes.csv", index=False)

data = Data(edge_index=edge_index)

pyg_data_to_edges_csv(data)
