"""
Loads real Drug→Disease cancer relationships using:
1. OpenTargets - known drugs for each cancer
2. MyDisease.io - correct drug endpoint
"""

import requests
import time
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    'bolt://localhost:7687', auth=('neo4j', 'password')
)

def insert_triple(head, relation, tail, head_label, tail_label):
    with driver.session() as session:
        session.run(f"""
            MERGE (h:{head_label} {{name: $head}})
            MERGE (t:{tail_label} {{name: $tail}})
            MERGE (h)-[r:{relation}]->(t)
        """, head=head, tail=tail)


def load_opentargets_drug_disease():
    """
    OpenTargets known drugs API — most reliable source.
    Returns approved drugs for each cancer type.
    """
    print("📡 Fetching Drug→Disease from OpenTargets Known Drugs...")

    cancer_diseases = [
        ("EFO_0001378", "Lung Cancer"),
        ("EFO_0000305", "Breast Cancer"),
        ("EFO_0000365", "Colorectal Cancer"),
        ("EFO_0000220", "Leukemia"),
        ("EFO_0000389", "Prostate Cancer"),
        ("EFO_0000616", "Lymphoma"),
        ("EFO_0000400", "Pancreatic Cancer"),
        ("EFO_0000558", "Kidney Cancer"),
        ("EFO_0001663", "Ovarian Cancer"),
        ("EFO_0000183", "Glioblastoma"),
        ("EFO_0000292", "Melanoma"),
        ("EFO_0001422", "Bladder Cancer"),
    ]

    url   = "https://api.platform.opentargets.org/api/v4/graphql"
    count = 0

    for disease_id, disease_name in cancer_diseases:
        query = """
        {
          disease(efoId: "%s") {
            knownDrugs(size: 30) {
              rows {
                drug {
                  name
                }
                phase
                status
              }
            }
          }
        }
        """ % disease_id

        try:
            response = requests.post(
                url, json={"query": query}, timeout=20
            )
            data = response.json()
            rows = (data.get("data", {})
                        .get("disease", {})
                        .get("knownDrugs", {})
                        .get("rows", []))

            for row in rows:
                drug_name = row.get("drug", {}).get("name", "")
                phase     = row.get("phase", 0)
                if drug_name and phase >= 3:  # only phase 3/4 approved drugs
                    insert_triple(
                        drug_name.title(), "TREATS",
                        disease_name, "Drug", "Disease"
                    )
                    count += 1

            print(f"  {disease_name}: {len(rows)} drugs")
            time.sleep(0.5)

        except Exception as e:
            print(f"  ⚠️ Skipped {disease_name}: {e}")

    print(f"✅ OpenTargets Drug→Disease: {count} triples loaded")
    return count


def load_ctd_drug_disease():
    """
    Comparative Toxicogenomics Database (CTD) — free API
    Returns curated drug-disease relationships for cancer.
    """
    print("\n📡 Fetching Drug→Disease from CTD...")

    cancer_mesh_ids = [
        ("D002289", "Lung Cancer"),
        ("D001943", "Breast Cancer"),
        ("D015179", "Colorectal Cancer"),
        ("D007938", "Leukemia"),
        ("D011471", "Prostate Cancer"),
        ("D008223", "Lymphoma"),
        ("D010190", "Pancreatic Cancer"),
        ("D007680",  "Kidney Cancer"),
        ("D010051", "Ovarian Cancer"),
        ("D005909", "Glioblastoma"),
        ("D008545", "Melanoma"),
    ]

    count = 0
    for mesh_id, disease_name in cancer_mesh_ids:
        url = (f"https://ctdbase.org/tools/batchQuery.go"
               f"?inputType=disease&inputTerms={mesh_id}"
               f"&report=chemicals_curated&format=json&limit=20")
        try:
            response = requests.get(url, timeout=20)
            data     = response.json()

            if isinstance(data, list):
                for item in data[:20]:
                    drug = item.get("ChemicalName", "")
                    if drug and len(drug) > 2:
                        insert_triple(
                            drug.title(), "TREATS",
                            disease_name, "Drug", "Disease"
                        )
                        count += 1

            print(f"  {disease_name}: inserted")
            time.sleep(0.4)

        except Exception as e:
            print(f"  ⚠️ Skipped {disease_name}: {e}")

    print(f"✅ CTD: {count} Drug→Disease triples loaded")
    return count


if __name__ == "__main__":
    print("🧬 Loading real Drug→Disease cancer data...\n")

    g1 = load_opentargets_drug_disease()
    g2 = load_ctd_drug_disease()

    with driver.session() as session:
        nodes = session.run(
            "MATCH (n) RETURN count(n) AS c").single()["c"]
        rels  = session.run(
            "MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        drug_dis = session.run(
            "MATCH (d:Drug)-[:TREATS]->(dis:Disease) "
            "RETURN count(*) AS c").single()["c"]

    driver.close()

    print(f"\n{'='*50}")
    print(f"🎉 Drug→Disease Loading Complete!")
    print(f"{'='*50}")
    print(f"  Total Nodes           : {nodes}")
    print(f"  Total Relationships   : {rels}")
    print(f"  Drug→Disease (TREATS) : {drug_dis}")
    print(f"  New triples added     : {g1 + g2}")