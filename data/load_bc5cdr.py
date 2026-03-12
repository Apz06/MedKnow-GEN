from datasets import load_dataset
from neo4j import GraphDatabase

# Load BC5CDR from HuggingFace
print('Downloading BC5CDR dataset...')
dataset = load_dataset('tner/bc5cdr', split='train')
print(f'Loaded {len(dataset)} samples')

# Connect to Neo4j
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))

# Extract unique chemicals and diseases
chemicals = set()
diseases  = set()

for sample in dataset:
    tokens = sample['tokens']
    tags   = sample['tags']
    current_entity = []
    current_tag    = None

    for token, tag in zip(tokens, tags):
        if tag == 3:
            if current_entity and current_tag == 'Chemical':
                chemicals.add(' '.join(current_entity))
            current_entity = [token]
            current_tag    = 'Chemical'
        elif tag == 4 and current_tag == 'Chemical':
            current_entity.append(token)
        elif tag == 1:
            if current_entity and current_tag == 'Disease':
                diseases.add(' '.join(current_entity))
            current_entity = [token]
            current_tag    = 'Disease'
        elif tag == 2 and current_tag == 'Disease':
            current_entity.append(token)
        else:
            if current_entity:
                if current_tag == 'Chemical':
                    chemicals.add(' '.join(current_entity))
                elif current_tag == 'Disease':
                    diseases.add(' '.join(current_entity))
            current_entity = []
            current_tag    = None

print(f'Found {len(chemicals)} chemicals and {len(diseases)} diseases')

# Load into Neo4j
with driver.session() as session:
    session.run('CREATE CONSTRAINT IF NOT EXISTS FOR (n:Drug) REQUIRE n.name IS UNIQUE')
    session.run('CREATE CONSTRAINT IF NOT EXISTS FOR (n:Disease) REQUIRE n.name IS UNIQUE')

    print('Inserting Drug nodes...')
    for chem in list(chemicals)[:500]:
        session.run('MERGE (d:Drug {name: $name})', name=chem)

    print('Inserting Disease nodes...')
    for dis in list(diseases)[:300]:
        session.run('MERGE (d:Disease {name: $name})', name=dis)

print('BC5CDR data loaded into Neo4j!')
print(f'Drugs: {min(len(chemicals), 500)} | Diseases: {min(len(diseases), 300)}')
driver.close()