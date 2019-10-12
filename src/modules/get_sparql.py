from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib import Graph

sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql.setQuery(
    """
construct { 
  <http://dbpedia.org/resource/Donald_Trump> ?p ?o .
  ?s ?p2 <http://dbpedia.org/resource/Donald_Trump> .
  } 
where { 
  { <http://dbpedia.org/resource/Donald_Trump> ?p ?y .
  ?y rdfs:label ?o} 
  union 
  { ?k ?p2 <http://dbpedia.org/resource/Donald_Trump> .
}
    FILTER(?p != <http://dbpedia.org/ontology/wikiPageWikiLink>) .
    FILTER(?p2 != <http://dbpedia.org/ontology/wikiPageWikiLink>) .
    FILTER (LANG(?o) = "en") .
}
"""
)

sparql.setReturnFormat(JSON)
results = sparql.query().convert()["results"]["bindings"]

# print(results)

for i in results:
    print(i["s"]["value"], i["p"]["value"], i["o"]["value"])
# g = Graph()
# g.parse(data=results, format="json")
# print(g.serialize(format='json'))
