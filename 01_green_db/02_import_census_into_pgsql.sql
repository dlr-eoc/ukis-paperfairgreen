DROP TABLE IF EXISTS census_bevoelkerung100m;
CREATE TABLE census_bevoelkerung100m (
  Gitter_ID_100m VARCHAR(16),
  Gitter_ID_100m_neu VARCHAR(30),
  Merkmal VARCHAR(50),
  Auspraegung_Code SMALLINT,
  Auspraegung_Text VARCHAR(60),
  Anzahl INT,
  Anzahl_q INT
);

\copy census_bevoelkerung100m from 'Bevoelkerung100M.csv' (delimiter ';', FORMAT CSV, HEADER, ENCODING 'latin1');


DROP TABLE IF EXISTS census_familie100m;
CREATE TABLE census_familie100m (
  Gitter_ID_100m VARCHAR(16),
  Gitter_ID_100m_neu VARCHAR(30),
  Merkmal VARCHAR(50),
  Auspraegung_Code SMALLINT,
  Auspraegung_Text VARCHAR(60),
  Anzahl INT,
  Anzahl_q INT
);

\copy census_familie100m from 'Familie100m.csv' (delimiter ',', FORMAT CSV, HEADER, ENCODING 'latin1');


DROP TABLE IF EXISTS census_haushalte100m;
CREATE TABLE census_haushalte100m (
  Gitter_ID_100m VARCHAR(16),
  Gitter_ID_100m_neu VARCHAR(30),
  Merkmal VARCHAR(50),
  Auspraegung_Code SMALLINT,
  Auspraegung_Text VARCHAR(60),
  Anzahl INT,
  Anzahl_q INT
);

\copy census_haushalte100m from 'Haushalte100m.csv' (delimiter ',', FORMAT CSV, HEADER, ENCODING 'latin1');
