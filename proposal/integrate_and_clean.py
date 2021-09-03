"""
Just a little file that runs both data_integrator.py and data_cleaner.py
so its easier
"""

from proposal import data_integrator, data_cleaner
import pandas as pd

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    di = data_integrator.data_integrator()
    di.assign_types_and_merge()

    dc = data_cleaner.data_cleaner()
    dc.clean()
