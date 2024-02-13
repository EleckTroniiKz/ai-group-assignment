import time
import json
import pandas as pd
import scipy.sparse as sparse
import implicit
from implicit import evaluation
from implicit.nearest_neighbours import bm25_weight
from itertools import product
from flask import Flask, request, jsonify, render_template


# Load user-item interaction matrix (R) from CSV file
print('Loading data...')
R_df = pd.read_csv(r"User-Item_Interaction_Matrix.csv", delimiter=';', index_col=0)
R_matrix = R_df.values

# Convert to a sparse matrix
sparse_user_item = sparse.csr_matrix(R_matrix)

"""
This Section is for testing the model with the best parameters

#Testing the Model
(train, test) = implicit.evaluation.train_test_split(sparse_user_item)
sparse_user_item_weighted = bm25_weight(train, K1=50, B=0.8)
model = implicit.als.AlternatingLeastSquares(factors=10,regularization=0.01,alpha=60)
trained = model.load(r"recommendation_model_ranked")
ranking = evaluation.ranking_metrics_at_k(trained,sparse_user_item_weighted,bm25_weight(test, K1=50, B=0.8))
print(str(ranking))
{'precision': 0.9032258064516129, 'map': 0.8833333333333333, 'ndcg': 0.9243462449410487, 'auc': 0.8322677021812552}
"""
# In this section the model is initated with the best parameter. As an alternative the model could be loaded from the
# recommendation_model_ranked file.

# #weighting the data
sparse_user_item_weighted = bm25_weight(sparse_user_item, K1=50, B=0.8)

# Initialize ALS model and fit using the sparse item-user matrix
model = implicit.als.AlternatingLeastSquares(factors=10, regularization=0.01, alpha=60)

# Generating new model with the complete data
model.fit(sparse_user_item_weighted)

model.save("recommendation_model")


# Flask app
app = Flask(__name__)


@app.route('/')
def index():
    # Open web browser e.g. http://127.0.0.1:5000/ to call index.html
    return render_template('index.html')


@app.route('/user_recommend', methods=['GET', 'POST'])
def user_recommend():
    if request.method == 'POST':
        try:
            # Get user ID from POST request
            user_id = int(request.form.get('user_id'))
            user_row_idx = list(R_df.index).index(user_id)

            # Generate recommendations using implicit model
            ids, scores = model.recommend(user_row_idx, sparse_user_item[user_row_idx])
            # Get Artwork IDs from Column Index
            artwork_ids = R_df.columns

            # Get explanation for the recommendations
            explanations = []
            for id in ids:
                explanation = model.explain(user_row_idx, sparse_user_item, id)
                try:
                    explanations.append(explanation[1][0])
                except:
                    print("Try another User-ID")

            # convert explanations to Artwork IDs
            explanations = [artwork_ids[explanation[0]] for explanation in explanations]
            # the profile information from each user is loaded. This information has been prepared in the transform
            # step in the file prepare_user_profile
            profile_dict = dict()
            with open("user_profile.txt", "r") as profile:
                # Load the dictionary from the file
                profile_dict = json.load(profile)
            print(profile_dict[str(user_id)])
            # Convert the results to a dictionary of artwork IDs and scores and explanations
            results = pd.DataFrame({'Artwork': artwork_ids[ids], 'Score': scores, 'Top Artworks': explanations})
            results_dict = results.set_index('Artwork').to_dict()
            #            results_df = pd.DataFrame(results_dict.items())

            #            return jsonify(results_dict)
            print(results_dict.items())
            return render_template('user_recommendation_form.html', data=(results_dict, profile_dict[str(user_id)]))

        except Exception as e:
            return jsonify({'error': str(e)})

    else:
        return render_template('user_recommendation_form.html')


@app.route('/artwork_recommend', methods=['GET', 'POST'])
def artwork_recommend():
    if request.method == 'POST':
        try:
            # Get artwork ID from the POST request
            artwork_id = request.form.get('artwork_id')
            artwork_col_idx = list(R_df.columns).index(artwork_id)

            # Generate recommendations using the implicit model
            ids, scores = model.similar_items(artwork_col_idx)

            # Get Artwork IDs from Column Index
            artwork_ids = R_df.columns

            results = pd.DataFrame({'Artwork': artwork_ids[ids], 'Score': scores})

            # Convert the results to a dictionary of artwork IDs and scores
            results_dict = results.set_index('Artwork').to_dict()['Score']
            print(results_dict)
            #            return jsonify(results_dict)
            return render_template('artwork_recommendation_form.html', data=results_dict)

        except Exception as e:
            return jsonify({'error': str(e)})

    else:
        return render_template('artwork_recommendation_form.html')


def find_parameters(train, test, num_iterations=5):
    """
    This method has been used to find the best parameters.
    """
    start = time.time()
    factors_values = [10, 20, 30, 40, 50]
    regularization_values = [0.01, 0.1, 0.2, 0.5]
    alpha_values = [0.1, 10, 40, 60]
    K1_values = [30, 50, 70, 100, 150, 200]
    B_values = [0.3, 0.5, 0.7, 0.8, 0.9]

    best_total = 0.0
    best_precision = 0.0
    best_map = 0.0
    best_ndcg = 0.0
    best_precision_params = None
    best_precision_metrics = None
    best_map_params = None
    best_map_metrics = None
    best_ndcg_params = None
    best_ndcg_metrics = None
    best_params = None
    best_metrics = None

    for _ in range(num_iterations):
        total_ranking = {'precision': 0.0, 'map': 0.0, 'ndcg': 0.0}

        for factor, regularization, alpha, K1, B in product(factors_values, regularization_values, alpha_values,
                                                            K1_values, B_values):
            weighted_train = bm25_weight(train, K1=K1, B=B)
            weighted_test = bm25_weight(test, K1=K1, B=B)
            untrained_model = implicit.als.AlternatingLeastSquares(factors=factor, regularization=regularization,
                                                                   alpha=alpha, num_threads=0)
            untrained_model.fit(weighted_train)
            ranking = evaluation.ranking_metrics_at_k(untrained_model, weighted_train, weighted_test,
                                                      show_progress=False, num_threads=0)
            total_ranking = ranking['precision'] + ranking['map'] + ranking['ndcg']
            if best_total < total_ranking:
                untrained_model.save("recommendation_model_ranked")
                best_total = total_ranking
                best_params = {'factor': factor, 'regularization': regularization, 'alpha': alpha, 'K1': K1, 'B': B}
                best_metrics = ranking

            if best_precision < ranking['precision']:
                best_precision = ranking['precision']
                best_precision_params = {'factor': factor, 'regularization': regularization, 'alpha': alpha, 'K1': K1,
                                         'B': B}
                best_precision_metrics = ranking

            if best_map < ranking['map']:
                best_map = ranking['map']
                best_map_params = {'factor': factor, 'regularization': regularization, 'alpha': alpha, 'K1': K1, 'B': B}
                best_map_metrics = ranking

            if best_ndcg < ranking['ndcg']:
                best_ndcg = ranking['ndcg']
                best_ndcg_params = {'factor': factor, 'regularization': regularization, 'alpha': alpha, 'K1': K1,
                                    'B': B}
                best_ndcg_metrics = ranking

    print("Best Total Ranking: ", best_total)
    print("Best parameters: ", best_params)
    print("Best metrics: ", best_metrics)
    print("Best precision: ", best_precision)
    print("Best precision Params: ", best_precision_params)
    print("Best precision Metrics: ", best_precision_metrics)
    print("Best map: ", best_map)
    print("Best map Params: ", best_map_params)
    print("Best map Metrics: ", best_map_metrics)
    print("Best NDCG: ", best_ndcg)
    print("Best NDCG Params: ", best_ndcg_params)
    print("Best NDCG Metrics: ", best_ndcg_metrics)
    end = start - time.time()
    print("Time elapsed: ", end)
    """
    Best Total Ranking:  0.6789080701155177
    Best parameters:  {'factor': 10, 'regularization': 0.01, 'alpha': 60, 'K1': 50, 'B': 0.8}
    Best metrics:  {'precision': 0.25, 'map': 0.18760912698412696, 'ndcg': 0.24129894313139066, 'auc': 0.542326237120426}
    Best precision:  0.25
    Best precision Params:  {'factor': 20, 'regularization': 0.01, 'alpha': 60, 'K1': 70, 'B': 0.8}
    Best precision Metrics:  {'precision': 0.25, 'map': 0.18357142857142855, 'ndcg': 0.23980020153771572, 'auc': 0.5424931605924342}
    Best map:  0.18760912698412696
    Best map Params:  {'factor': 10, 'regularization': 0.01, 'alpha': 60, 'K1': 50, 'B': 0.8}
    Best map Metrics:  {'precision': 0.25, 'map': 0.18760912698412696, 'ndcg': 0.24129894313139066, 'auc': 0.542326237120426}
    Best NDCG:  0.24129894313139066
    Best NDCG Params:  {'factor': 10, 'regularization': 0.01, 'alpha': 60, 'K1': 50, 'B': 0.8}
    Best NDCG Metrics:  {'precision': 0.25, 'map': 0.18760912698412696, 'ndcg': 0.24129894313139066, 'auc': 0.542326237120426}
    Time elapsed:  -1544.31663107872
    We have chosen the following values for the best parameters: {'factor': 10, 'regularization': 0.01, 'alpha': 60, 'K1': 50, 'B': 0.8}.
    """


if __name__ == '__main__':
    app.run(debug=False)  # Set to False for production
