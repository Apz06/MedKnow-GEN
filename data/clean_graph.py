from neo4j import GraphDatabase

driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))

with driver.session() as session:
    result = session.run('''
        MATCH (d:Drug)
        WHERE size(d.name) < 3
           OR d.name IN ['pain', 'with', 'basal', 'cascade', 
                         'cornu', 'tolerance', 'abnormalities',
                         'anaemia', 'tinnitus']
        DETACH DELETE d
        RETURN count(d) AS deleted
    ''')
    record = result.single()
    print(f'Deleted {record["deleted"]} noise nodes')

driver.close()
print('Done!')