from sklearn.feature_extraction.text import TfidfVectorizer
import nltk.stem
import sklearn.datasets
from sklearn.cluster import KMeans
import scipy as sp 

def load_data():
    training_data = sklearn.datasets.fetch_20newsgroups(subset='train')
    testing_data = sklearn.datasets.fetch_20newsgroups(subset='test')
    return training_data, testing_data

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
      analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
      return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

def preprocess_data(training_data, vectorizer):
    vectorized_data = vectorizer.fit_transform(training_data.data)
    return vectorized_data

def KMeans_train(vectorized_data, num_of_cluster):
    km = KMeans(n_clusters=num_of_cluster, init='random', n_init=1, verbose=1, random_state=3)
    km.fit(vectorized_data)
    return km

def get_related_posts(query, dataset, vectorizer, num_of_related_post):
    vectorized_data = preprocess_data(dataset, vectorizer)
    query_vec = vectorizer.transform([query])
    km = KMeans_train(vectorized_data, 20)
    query_label = km.predict(query_vec)[0]
    similar_indices = (km.labels_ == query_label).nonzero()[0]
    similar_post = []
    for i in similar_indices:
        dist = sp.linalg.norm((query_vec - vectorized_data[i]).toarray())
        similar_post.append((dist, dataset.data[i]))
    similar_post = sorted(similar_post)
    return similar_post[:num_of_related_post]

english_stemmer = nltk.stem.SnowballStemmer('english')
vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5, stop_words='english', decode_error='ignore')
training_data, testing_data = load_data()
query = "Disk drive problems. Hi, I have a problem with my hard disk. After 1 year it is working only sporadically now. I tried to format it. but now it doesn't boot any more. Any ideas? Thanks."
similar_posts = get_related_posts(query, training_data, vectorizer, 5)
for idx, post in enumerate(similar_posts, 1):
    print ("===== Post %d =====" % idx)
    print ("Similarity: %f" % post[0])
    print (post[1])

