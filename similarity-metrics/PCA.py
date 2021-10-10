"""
Takes the movie_df_in_pandas_form.pickle.lz4 and runs through all of the steps
again
"""
import numpy as np

from useful_tools import pickle_manager
from sklearn.decomposition import PCA
import pandas as pd

from useful_tools import pickle_manager


def main():
    vector_features: pd.DataFrame = pickle_manager.load_pickle(
        '../pickles/sentence_features.pickle.lz4'
    )
    vector_features = vector_features.reset_index().to_numpy(dtype=np.float64)

    # Huh apparently we have nan. If we set to 0 then it should be fine
    vector_features[np.isnan(vector_features)] = 0

    pca = PCA(n_components=2)
    print(pca.fit_transform(vector_features))


if __name__ == "__main__":
    main()
