from __future__ import annotations

import unittest

import pandas as pd

from compressor_ml.mysql_source import MySQLReadingsSource, MySQLSourceConfig


def source(sensor_scale: str = "auto") -> MySQLReadingsSource:
    return MySQLReadingsSource(
        MySQLSourceConfig(
            host="10.195.17.73",
            port=3306,
            database="ht9046mx_iot",
            user_env="MYSQL_USER",
            password_env="MYSQL_PASSWORD",
            readings_table="ht9046mx_readings",
            machine_column="machine_number",
            timestamp_column="recorded_at",
            module_column=None,
            timezone="Asia/Bangkok",
            sensor_scale=sensor_scale,
            connect_timeout_seconds=10,
            read_timeout_seconds=300,
            max_rows_per_query=1000,
        )
    )


class MySQLCanonicalizationTests(unittest.TestCase):
    def test_wide_handler_columns_become_one_canonical_row_per_module(self):
        raw = pd.DataFrame(
            {
                "recorded_at": ["2026-08-20 08:00:00"],
                "Status": ["Run"],
                "Hp_1st_1": [1200],
                "Lp_1st_1": [300],
                "Hp_2nd_1": [2100],
                "Lp_2nd_1": [520],
                "Valve_1": [60],
                "TempHi_1": [700],
                "TempLo_1": [200],
                "Status_1": ["On"],
                "Busy_1": [0],
                "SV_1": ["On"],
            }
        )
        result = source().canonicalize(raw, "57")
        self.assertEqual(result.iloc[0]["machine_id"], "MX057")
        self.assertEqual(int(result.iloc[0]["module_id"]), 1)
        self.assertEqual(result.iloc[0]["global_status"], "Run")
        self.assertEqual(result.iloc[0]["module_status"], "On")
        self.assertEqual(float(result.iloc[0]["hp2"]), 210.0)
        self.assertEqual(float(result.iloc[0]["temphi"]), 70.0)
        self.assertEqual(str(result.iloc[0]["timestamp"].tz), "Asia/Bangkok")

    def test_long_rows_are_normalized_without_wide_column_suffixes(self):
        raw = pd.DataFrame(
            {
                "recorded_at": ["2026-08-20 08:00:00"],
                "module_id": [2],
                "status": ["Run"],
                "module_status": ["On"],
                "busy": [0],
                "sv": ["On"],
                "hp1": [120],
                "lp1": [30],
                "hp2": [210],
                "lp2": [52],
                "valve": [60],
                "temphi": [70],
                "templo": [20],
            }
        )
        result = source("1").canonicalize(raw, "MX057")
        self.assertEqual(len(result), 1)
        self.assertEqual(int(result.iloc[0]["module_id"]), 2)
        self.assertEqual(float(result.iloc[0]["lp2"]), 52.0)


if __name__ == "__main__":
    unittest.main()
