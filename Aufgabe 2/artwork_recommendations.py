import pandas as pd
import scipy.sparse as sparse
import implicit
from implicit.nearest_neighbours import bm25_weight

# Load user-item interaction matrix (R) from CSV file
print('Loading data...')
R_df = pd.read_csv(r"Artwork_Recommendations_TIF21\User-Item_Interaction_Matrix_Template_With_Randomized_Values.csv", delimiter=';', index_col=0)

R_matrix = R_df.values

# Convert to a sparse matrix
sparse_user_item = sparse.csr_matrix(R_matrix)

#Refine interaction matrix with weighting scheme before training
sparse_user_item_weighted = bm25_weight(sparse_user_item, K1=70, B=0.8)

# Initialize ALS model and fit using the sparse item-user matrix
model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, alpha=40)

# Fit the model
model.fit(sparse_user_item_weighted)

#save the model in local file storage
model.save(r"Artwork_Recommendations_TIF21\recommendation_model")



#Flask app
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def index():
    #Open web browser e.g. http://127.0.0.1:5000/ to call index.html
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
            
            #convert explanations to Artwork IDs
            explanations = [artwork_ids[explanation[0]] for explanation in explanations]

            # Convert the results to a dictionary of artwork IDs and scores and explanations
            results = pd.DataFrame({'Artwork': artwork_ids[ids], 'Score': scores, 'Top Artworks': explanations})
            results_dict = results.set_index('Artwork').to_dict()
#            results_df = pd.DataFrame(results_dict.items())

#            return jsonify(results_dict)
            return render_template('user_recommendation_form.html', data=results_dict)

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

#            return jsonify(results_dict)
            return render_template('artwork_recommendation_form.html', data=results_dict)

        except Exception as e:
            return jsonify({'error': str(e)})

    else:
        return render_template('artwork_recommendation_form.html')

if __name__ == '__main__':
    app.run(debug=False)  # Set to False for production