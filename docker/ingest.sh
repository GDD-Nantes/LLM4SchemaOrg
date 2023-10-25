#!/bin/bash
input_nq=$1

docker exec docker-schemamarkup-virtuoso-1 /opt/virtuoso-opensource/bin/isql "EXEC=grant select on \"DB.DBA.SPARQL_SINV_2\" to \"SPARQL\";"
docker exec docker-schemamarkup-virtuoso-1 /opt/virtuoso-opensource/bin/isql "EXEC=grant execute on \"DB.DBA.SPARQL_SINV_IMP\" to \"SPARQL\";"

docker exec docker-schemamarkup-virtuoso-1 /opt/virtuoso-opensource/bin/isql "EXEC=ld_dir('/usr/share/proj/', '$input_nq', 'http://example.com/datasets/default');" >> /dev/null

docker exec docker-schemamarkup-virtuoso-1 /opt/virtuoso-opensource/bin/isql "EXEC=rdf_loader_run(log_enable=>2);" &&
docker exec docker-schemamarkup-virtuoso-1 /opt/virtuoso-opensource/bin/isql "EXEC=checkpoint;"&&
exit 0
