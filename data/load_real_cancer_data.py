def load_chembl_drug_targets():
    print("\n📡 Fetching Drug→Gene from ChEMBL...")

    # Use ChEMBL IDs directly for cancer drugs (more reliable)
    cancer_drug_ids = {
        "Gefitinib":   "CHEMBL939",
        "Erlotinib":   "CHEMBL553",
        "Imatinib":    "CHEMBL941",
        "Sorafenib":   "CHEMBL1336",
        "Sunitinib":   "CHEMBL535",
        "Tamoxifen":   "CHEMBL83",
        "Trastuzumab": "CHEMBL1201585",
        "Bevacizumab": "CHEMBL1201583",
        "Doxorubicin": "CHEMBL53463",
        "Vincristine": "CHEMBL303560",
        "Rituximab":   "CHEMBL1201576",
        "Bortezomib":  "CHEMBL325041",
        "Paclitaxel":  "CHEMBL428647",
        "Docetaxel":   "CHEMBL92",
        "Fluorouracil":"CHEMBL185",
        "Oxaliplatin": "CHEMBL414804",
        "Pemetrexed":  "CHEMBL1201236",
    }

    count = 0
    for drug_name, chembl_id in cancer_drug_ids.items():
        url = (f"https://www.ebi.ac.uk/chembl/api/data/mechanism"
               f"?molecule_chembl_id={chembl_id}&format=json&limit=10")
        try:
            response = requests.get(url, timeout=20)
            data     = response.json()
            mechs    = data.get("mechanisms", [])

            for mech in mechs:
                # Extract target from mechanism_of_action
                # e.g. "EGFR inhibitor" → "EGFR"
                moa    = mech.get("mechanism_of_action", "")
                action = mech.get("action_type", "TARGETS")

                # Extract gene/target name (first word(s) before inhibitor/activator)
                target = moa.replace(" inhibitor", "")\
                            .replace(" INHIBITOR", "")\
                            .replace(" activator", "")\
                            .replace(" antagonist", "")\
                            .replace(" agonist", "")\
                            .strip()

                if target and 2 < len(target) < 50 and mech.get("disease_efficacy") == 1:
                    relation = "INHIBITS" if "INHIBIT" in str(action) else "TARGETS"
                    insert_triple(drug_name, relation, target, "Drug", "Gene")
                    count += 1

            if mechs:
                print(f"  {drug_name}: {len(mechs)} mechanisms → inserted")
            time.sleep(0.4)

        except Exception as e:
            print(f"  ⚠️ Skipped {drug_name}: {e}")

    print(f"✅ ChEMBL: {count} Drug→Gene triples loaded")
    return count


def load_mydisease_drug_disease():
    print("\n📡 Fetching Drug→Disease from MyDisease.io...")

    cancer_terms = [
        "breast cancer", "colorectal cancer", "leukemia",
        "lymphoma", "prostate cancer", "melanoma",
        "glioblastoma", "pancreatic cancer", "ovarian cancer",
        "bladder cancer", "kidney cancer",
    ]

    count = 0
    for term in cancer_terms:
        url = f"https://mydisease.info/v1/query?q={term}&fields=pharmgkb&size=20"
        try:
            response = requests.get(url, timeout=20)
            data     = response.json()
            hits     = data.get("hits", [])
            disease_name = term.title()

            for hit in hits:
                pgkb = hit.get("pharmgkb", {})
                if not pgkb:
                    continue
                drugs = pgkb.get("chemicals", [])
                if isinstance(drugs, list):
                    for d in drugs[:5]:
                        name = d.get("name", "") if isinstance(d, dict) else str(d)
                        if name and len(name) > 2:
                            insert_triple(name.title(), "TREATS",
                                         disease_name, "Drug", "Disease")
                            count += 1

            print(f"  {disease_name}: {len(hits)} hits")
            time.sleep(0.3)

        except Exception as e:
            print(f"  ⚠️ Skipped {term}: {e}")

    print(f"✅ MyDisease.io: {count} Drug→Disease triples loaded")
    return count