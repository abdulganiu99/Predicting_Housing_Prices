import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# To make the transformer more robust, we hard-code the column indices.
# These correspond to the positions of the columns in the original DataFrame:
# total_rooms=3, total_bedrooms=4, population=5, households=6
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    A custom transformer to add combined attributes. This version is designed
    to work seamlessly within a Scikit-Learn pipeline by operating on column
    indices, which are preserved when data is converted to a NumPy array.
    """
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        # Calculate new attributes using the predefined column indices
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

    def get_feature_names_out(self, input_features=None):
        """
        Generates output feature names for the transformed data.
        This is crucial for pipelines to track column names.
        """
        # Start with the input feature names
        output_features = list(input_features)
        # Add the names of the new features
        output_features.extend(["rooms_per_household", "population_per_household"])
        if self.add_bedrooms_per_room:
            output_features.append("bedrooms_per_room")
        return np.array(output_features)
