#!/bin/bash
docker exec docker-schemamarkup-virtuoso-1 /opt/virtuoso-opensource/bin/isql "EXEC=dump_nquads ('/usr/share/proj/dump', 1, 10000000000, 1);"
