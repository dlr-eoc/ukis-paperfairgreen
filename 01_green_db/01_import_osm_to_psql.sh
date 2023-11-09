#!/bin/bash

# start postgis docker
sudo docker-compose up

# create database
# psql -h localhost -p 5432 -U u -d green_lu -c "CREATE DATABASE osm;"

# enable extensions
# psql -h localhost -p 5432 -U u -d osm -c "CREATE DATABASE postgis;"
# psql -h localhost -p 5432 -U u -d osm -c "CREATE EXTENSION hstore;"

# import osm.pbf file to postgis
osm2pgsql -d green_lu \
	  -U u \
	  -W \
	  -H localhost \
	  -P 5432 \
	  -C 25000 \
	  -s --drop \
	  -p osm \
	  --hstore --hstore-add-index \
	  -G \
	  --number-processes=12 \
	  -E 3035 \
	  2021-01-06_germany-latest.osm.pbf

