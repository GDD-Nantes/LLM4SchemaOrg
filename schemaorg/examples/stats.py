from rdflib import ConjunctiveGraph

g = ConjunctiveGraph()
g.parse("schemaorg-all-examples.ttl", format="turtle")

QUERY = """
SELECT 
    ?entry 
    (COUNT(?text) AS ?nbText) 
    (COUNT(?rdfa) AS ?nbRdfa) 
    (COUNT(?microdata) AS ?nbMicrodata) 
    (COUNT(?jsonld) AS ?nbJsonld)
WHERE {
    ?entry <http://example.org/hasExample> ?ex .
    OPTIONAL { ?ex <http://example.org/pre-markup> ?text . }
    OPTIONAL { ?ex <http://example.org/microdata> ?microdata . }
    OPTIONAL { ?ex <http://example.org/rdfa> ?rdfa . }
    OPTIONAL { ?ex <http://example.org/json> ?jsonld . }
} GROUP BY ?entry
"""

g.query(QUERY).serialize("stats.csv", format="csv")