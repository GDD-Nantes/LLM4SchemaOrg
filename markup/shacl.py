from datetime import datetime
from urllib.parse import quote
import click
from rdflib import OWL, RDF, RDFS, SH, XSD, BNode, ConjunctiveGraph, Literal, Namespace, URIRef
from tqdm import tqdm
from utils import schema_simplify

@click.group
def cli():
    pass

schema = Namespace("http://schema.org/")
datashapes = Namespace("http://datashapes.org/")

@cli.command()
@click.argument("infile", type=click.Path(exists=True,file_okay=True, dir_okay=False))
@click.argument("outfile", type=click.Path(file_okay=True, dir_okay=False))
def generate_shacl_shape(infile, outfile):
    """Generate SHACL shape similar to https://datashapes.org/schema
    Notes:
    - domainIncludes: property 
    - rangeIncludes: expected_type
    - nodes of type rdfs:Class is a Node
    """
    
    in_graph = ConjunctiveGraph()
    in_graph.parse(infile)

    out_graph = ConjunctiveGraph()
    out_graph.namespace_manager.bind("schema1", schema, replace=True)
    out_graph.namespace_manager.bind("ds", datashapes, replace=True)
    
    # Prelude
    out_graph.add((datashapes.schema, RDF.type, OWL.Ontology))
    out_graph.add((datashapes.schema, RDFS.comment, Literal("<p>This is an RDF/SHACL version of schema.org, generated based on the official Turtle file https://schema.org/version/latest/schemaorg-all-http.ttl. Alignments with common RDF practices have been made, e.g. using rdfs:Class instead of schema:Class.</p><p>Inspired by the work of Holger Knublauch</p>", datatype=RDF.HTML)))
    out_graph.add((datashapes.schema, RDFS.label, Literal("Schema.org SHACL shapes")))
    out_graph.add((datashapes.schema, OWL.imports, datashapes.dash))
    
    current_timestamp = datetime.now().isoformat()
    out_graph.add((datashapes.schema, OWL.versionInfo, Literal(current_timestamp, datatype=XSD.dateTime)))
    
    def create_bnode(p, o):
        bnode = BNode()
        out_graph.add((bnode, p, o))
        return bnode
    
    for node, node_class in tqdm(in_graph.subject_objects(RDF.type)):         
        if node_class == RDF.Property:
            prop = node
            for domain_node in in_graph.objects(prop, schema.domainIncludes):
                domain_simple = schema_simplify(domain_node)
                prop_simple = schema_simplify(prop)
                prop_node = URIRef(f"http://schema.org/{domain_simple}-{prop_simple}")
                
                out_graph.add((prop_node, RDF.type, SH.PropertyShape))
                out_graph.add((domain_node, SH.property, prop_node))
                out_graph.add((prop_node, SH.path, prop))
            
                # Copy class infos
                for prop_pred, prop_object in in_graph.predicate_objects(prop):
                    if prop_pred == schema.rangeIncludes: # expected type
                        collection_list = set()
                        
                        for expected_type in in_graph.objects(prop, schema.rangeIncludes):
                            
                            if str(expected_type).startswith(str(schema)):
                                collection_list.add((SH.datatype, XSD.string))
                            
                            if expected_type == schema.Text:
                                collection_list.add((SH.datatype, XSD.string))
                            elif expected_type in [schema.Number, schema.Float, schema.Integer]:
                                collection_list.update([
                                    (SH.datatype, XSD.float),
                                    (SH.datatype, XSD.double), 
                                    (SH.datatype, XSD.integer)
                                ])
                            elif expected_type == schema.Date:
                                collection_list.update([
                                    (SH.datatype, XSD.date),
                                    (SH.datatype, schema.Date)
                                ])
                            elif expected_type == schema.DateTime:
                                collection_list.update([
                                    (SH.datatype, XSD.dateTime),
                                    (SH.datatype, schema.DateTime)
                                ])
                            elif expected_type == schema.Time:
                                collection_list.update([
                                    (SH.datatype, XSD.time),
                                    (SH.datatype, schema.Time)
                                ])
                            elif expected_type == schema.Boolean:
                                collection_list.update([
                                    (SH.datatype, XSD.boolean),
                                    (SH.datatype, schema.Boolean)
                                ])
                            elif expected_type == schema.URL:
                                collection_list.update([
                                    (SH.nodeKind, SH.IRI)
                                ])
                            else:
                                collection_list.add((SH["class"], expected_type))
                        if len(collection_list) == 1:
                            p, o = next(iter(collection_list))
                            if (prop_node, p, o) not in out_graph:
                                out_graph.add((prop_node, p, o))
                        else:
                            if len(list(out_graph.objects(prop_node, SH["or"]))) == 0:
                                collection = out_graph.collection(BNode())
                                collection += [ create_bnode(*col) for col in collection_list ]
                                out_graph.add((prop_node, SH["or"], collection.uri))    
                    elif prop_pred == schema.domainIncludes:
                        continue
                    else:
                        out_graph.add((prop_node, prop_pred, prop_object))
                    out_graph.remove((prop_node, RDF.type, RDF.Property))
        else:
            if node_class == RDFS.Class:
                out_graph.add((node, RDF.type, SH.NodeShape))

            # Copy class infos
            for node_pred, node_object in in_graph.predicate_objects(node):
                out_graph.add((node, node_pred, node_object))
                
    out_graph.add((schema.Thing, RDFS.subClassOf, OWL.Thing))
    out_graph.serialize(outfile, format="turtle")
    
    # Epilogue: prepend import lines 
    with open(outfile, "r+") as f:
        lines = f.readlines()
        f.seek(0)
        lines.insert(0, "# baseURI: http://datashapes.org/schema\n")
        lines.insert(1, "# imports: http://datashapes.org/dash\n")
        lines.insert(2, "\n")
        f.writelines(lines)
    
@cli.command()
@click.argument("infile", type=click.Path(exists=True,file_okay=True, dir_okay=False))
@click.argument("outfile", type=click.Path(file_okay=True, dir_okay=False))
@click.option("--add", type=click.Path(exists=True, file_okay=True, dir_okay=False))
def close_schemaorg_ontology(infile, outfile, add):
    """Load an input SHACL shape graph and close each shape 
    by bringing all property from parent class to currend class shape 
    then add sh:closed at the end

    Args:
        infile (_type_): _description_
        outfile (_type_): _description_
        add (_type_): _description_
    """     
    
    graph = ConjunctiveGraph()    
    graph.parse(infile, format="turtle")
    
    if add:
        graph.parse(add, format="turtle")
             
    query = f"""
    SELECT DISTINCT ?shape ?parentShape ?parentProp WHERE {{
        ?shape  a <http://www.w3.org/ns/shacl#NodeShape> ;
                a <http://www.w3.org/2000/01/rdf-schema#Class> ;
                <http://www.w3.org/2000/01/rdf-schema#subClassOf>* ?parentShape .
                
        ?parentShape <http://www.w3.org/ns/shacl#property> ?parentProp .
        FILTER(?parentShape != ?shape)
    }}
    """ 
    
    results = graph.query(query)
    visited_shapes = set()
    for result in tqdm(results):
        shape = result.get("shape")
        parent_prop = result.get("parentProp")
        parent_shape = result.get("parentShape")
        graph.add((shape, SH.property, parent_prop))
        graph.add((shape, RDF.type, parent_shape))
        graph.add((shape, RDFS.subClassOf, parent_shape))
        graph.add((shape, SH.closed, Literal(True)))
        
        # subj sh:ignoredProperties ( rdf:type owl:sameAs )
        # https://www.w3.org/TR/turtle/#collections
        if shape not in visited_shapes:
            ignored_props = graph.collection(BNode())
            ignored_props += [RDF.type, OWL.sameAs]
            
            graph.add((shape, SH.ignoredProperties, ignored_props.uri))
            visited_shapes.add(shape)
    
    # Replace xsd:float with xsd:double
    for prop, value in tqdm(graph.subject_objects(SH.datatype)):
        if value == XSD.float:
            graph.set((prop, SH.datatype, XSD.double))
        elif value == XSD.date:
            graph.set((prop, SH.datatype, schema.Date))
        elif value == XSD.dateTime:
            graph.set((prop, SH.datatype, schema.DateTime))
    
    graph.serialize(outfile, format="turtle")
        
if __name__ == "__main__":
    cli()
    

