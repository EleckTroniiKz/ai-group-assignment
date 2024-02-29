import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the test dataset from the text file
test_dataset = pd.read_csv("captions.txt")

# Load the saved captions from BLIP model
blip_results = pd.read_csv("blip_results.csv")

# Load the saved captions from nlp_connect_result model
nlp_connect_results = pd.read_csv("nlp_connect_result.csv")

# Initialize an empty DataFrame to store the comparison results
comparison_results = pd.DataFrame(columns=['pictureid', 'actualdescription', 'blip_similarity', 'nlp_connect_similarity'])

vectorizer = TfidfVectorizer(stop_words='english')

# Combine all captions for BLIP and nlp_connect_result models for vectorization
blip_captions = blip_results['Caption']
nlp_connect_captions = nlp_connect_results['Caption']
all_captions = list(test_dataset['caption']) + list(blip_captions) + list(nlp_connect_captions)

# Fit the vectorizer and transform captions to TF-IDF representation
X = vectorizer.fit_transform(all_captions)

# Get the TF-IDF representations for test dataset captions
test_dataset_tfidf = X[:len(test_dataset)]
blip_tfidf = X[len(test_dataset):len(test_dataset)+len(blip_captions)]
nlp_connect_tfidf = X[len(test_dataset)+len(blip_captions):]

# Compute pairwise cosine similarity between test dataset captions and BLIP/nlp_connect_result captions
blip_similarities = cosine_similarity(test_dataset_tfidf, blip_tfidf)
nlp_connect_similarities = cosine_similarity(test_dataset_tfidf, nlp_connect_tfidf)
blip_average_similarity = blip_similarities.mean()
nlp_connect_average_similarity = nlp_connect_similarities.mean()

print("Average Similarity for BLIP Model:", blip_average_similarity)
print("Average Similarity for nlp_connect_result Model:", nlp_connect_average_similarity)

# Fill comparison_results DataFrame
for idx, row in test_dataset.iterrows():
    picture_id = row["image"]
    actual_description = row["caption"]
    blip_similarity = blip_similarities[idx].max()  # Max similarity across BLIP captions
    nlp_connect_similarity = nlp_connect_similarities[idx].max()  # Max similarity across nlp_connect_result captions
    new_row = pd.DataFrame(
        {'pictureid': picture_id, 'actualdescription': actual_description, 'blip_similarity': blip_similarity,
         'nlp_connect_similarity': nlp_connect_similarity}, index=[0])
    comparison_results = pd.concat([comparison_results, new_row], ignore_index=True)

# Save the comparison results DataFrame to a CSV file
comparison_results.to_csv("comparison_results_tfidf_with_stopwords.csv", index=False)
