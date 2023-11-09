from configparser import ConfigParser
from copy import deepcopy

import psycopg2
from geopandas import GeoSeries, clip, read_postgis
from key_values import KeyValues
from pandas import DataFrame, merge
from shapely.geometry import box


class GreenluDBhandler:
    def __init__(self, config_file, chunk_extent):
        "create data base object from config file"

        self.parser = ConfigParser()
        self.parser.read(config_file)
        self.db_param = {}
        if self.parser.has_section("postgresql"):
            params = self.parser.items("postgresql")
            for p in params:
                self.db_param[p[0]] = p[1]
        else:
            raise Exception(f"Section postgresql not found in the {config_file} file")

        self.con = psycopg2.connect(**self.db_param)

        # create key values instance
        self.kvo = KeyValues("data/mapfeatures.csv")

        self.prepare_chunk(chunk_extent)

    def __del__(self):
        """Destruct GreenluDB objects."""
        if self.con is not None:
            self.con.close()

    def prepare_chunk(self, chunk_extent):
        self.construct_queries(chunk_extent)
        self.gpd_eua = self.get_query_gpd(self.query_eua, "geom")
        self.gpd_point = self.get_query_gpd(self.query_point, "way")
        self.gpd_line = self.get_query_gpd(self.query_line, "way")
        self.gpd_polygon = self.get_query_gpd(self.query_polygon, "way")

    def get_query(self, query):
        """get query from database"""
        cur = self.con.cursor()
        cur.execute(query)
        return cur.fetchall()

    def get_query_gpd(self, query, geom_col):
        """Send query to database and return values"""
        the_gpd = read_postgis(query, self.con, geom_col=geom_col)
        if the_gpd.crs is None:
            the_gpd = the_gpd.set_crs(3035)
        return the_gpd

    def _test_con(self):
        """test the connection"""
        cur = self.con.cursor()
        cur.execute("SELECT version();")
        print(cur.fetchall())
        cur.close()

    def construct_queries(self, extent):
        """construct the query using the given extent"""
        xmin, ymin, xmax, ymax, *dump = extent

        ext_pg = f"""'SRID=3035;POLYGON(({xmin} {ymin}, {xmax} {ymin},
                                         {xmax} {ymax}, {xmin} {ymax}, {xmin} {ymin}))'"""

        def concat_query(table, filter_str, filter_extent):
            return f"SELECT * FROM {table} WHERE {filter_str} AND {filter_extent};"

        # construct eua query
        filter_extent = f"ST_INTERSECTS(geom,{ext_pg})"
        self.query_eua = f"SELECT * FROM eua_de_tiled WHERE {filter_extent};"

        # construct osm queries
        filter_extent = f"ST_INTERSECTS(way,{ext_pg})"
        self.query_point = concat_query("osm_point", self.kvo.filter_point, filter_extent)
        self.query_line = concat_query("osm_line", self.kvo.filter_line, filter_extent)
        self.query_polygon = concat_query("osm_polygon", self.kvo.filter_polygon, filter_extent)

    def add_kv_column(self, df_long, geom_type):
        kv = self.kvo.get_kv("kv", geom_type)
        return merge(df_long, kv, on=["key", "value"], how="inner")

    def pivot_longer_kv(self, df, id_vars, kv_geom):
        df_long = df.melt(
            id_vars=id_vars,
            value_vars=list(set(self.kvo.get_kv("k", kv_geom))),
            var_name="key",
            value_name="value",
        )
        return df_long.loc[df_long.loc[:, "value"].notnull(), :]

    def aggregate_features_in_window(self, extent):
        """Aggregate tiled polygons to useful information"""
        ext_gpd = GeoSeries(box(*extent), crs="EPSG:3035")

        # eua polygons
        tmp_gpd_eua = clip(self.gpd_eua.copy(), ext_gpd)
        tmp_gpd_eua["area_m2"] = tmp_gpd_eua.loc[:, "geom"].area
        tmp_gpd_eua["group"] = [self.kvo.kv_eua[str(x)] for x in tmp_gpd_eua.loc[:, "code_2018"]]
        tmp_df_eua = tmp_gpd_eua.loc[:, ("group", "area_m2")].groupby("group").agg("sum")
        tmp_df_eua["name"] = tmp_df_eua.index.values
        template = deepcopy(self.kvo.eua_dict_template)
        for v, k in tmp_df_eua.values.tolist():
            template[k] = v
        # replace None with 0
        tmp_agg_eua = {k: 0 if not v else v for k, v in template.items()}

        # osm polygons
        tmp_gpd_pol = clip(self.gpd_polygon.copy(), ext_gpd)
        # calculate area
        tmp_gpd_pol["area_m2"] = tmp_gpd_pol.loc[:, "way"].area
        # convert to a data frame and drop geometry
        df = DataFrame(tmp_gpd_pol.drop(columns="way"))
        # pivot longer
        df_long = self.pivot_longer_kv(df, ["osm_id", "area_m2"], "polygon")
        df_long = self.add_kv_column(df_long, "polygon")
        tmp_agg_polygon = df_long.loc[:, ("group", "area_m2")].groupby("group").agg("sum")

        # osm lines
        tmp_gpd_lin = clip(self.gpd_line.copy(), ext_gpd)
        # calculate line length
        tmp_gpd_lin["length_m"] = tmp_gpd_lin.loc[:, "way"].length
        # convert to a data frame and drop geometry
        df = DataFrame(tmp_gpd_lin.drop(columns="way"))
        # pivot longer
        df_long = self.pivot_longer_kv(df, ["osm_id", "length_m"], "line")
        df_long = self.add_kv_column(df_long, "line")
        tmp_agg_line = df_long.loc[:, ("group", "length_m")].groupby("group").agg("sum")

        # osm points
        tmp_gpd_poi = clip(self.gpd_point.copy(), ext_gpd)
        df = DataFrame(tmp_gpd_poi.drop(columns="way"))
        # pivot longer
        df_long = self.pivot_longer_kv(df, ["osm_id"], "point")
        df_long = self.add_kv_column(df_long, "point")
        tmp_agg_point = df_long.loc[:, ("group", "osm_id")].groupby("group").agg("count")

        return [self.join_osm_aggregates(tmp_agg_polygon, tmp_agg_line, tmp_agg_point), tmp_agg_eua]

    def join_osm_aggregates(self, agg_polygon, agg_line, agg_point):
        template = deepcopy(self.kvo.kv_dict_template)
        if len(agg_polygon.index.values) > 0:
            agg_polygon["name"] = "polygon_" + agg_polygon.index.values
        if len(agg_line.index.values) > 0:
            agg_line["name"] = "line_" + agg_line.index.values
        if len(agg_point.index.values) > 0:
            agg_point["name"] = "point_" + agg_point.index.values

        for v, k in (
            agg_polygon.values.tolist() + agg_line.values.tolist() + agg_point.values.tolist()
        ):
            template[k] = v

        # replace None with 0
        return {k: 0 if not v else v for k, v in template.items()}
