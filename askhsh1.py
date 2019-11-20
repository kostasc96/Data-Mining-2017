import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn import cluster
from wordcloud import WordCloud, STOPWORDS
from scipy.misc import imread
import matplotlib.pyplot as plt
from nltk.cluster import KMeansClusterer, cosine_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

train_data = pd.read_csv('train_set.csv', sep='\t')
test_data = pd.read_csv('test_set.csv', sep='\t')
expected_array = np.array(train_data[['Category']]).flatten()
content_array = np.array(train_data[['Content']]).flatten()

def word_cloud():
    politics = str()
    film = str()
    football = str()
    business = str()
    technology = str()
    for i in range(expected_array.shape[0]):
        if expected_array[i] == 'Politics':
            politics = politics + ' ' + str(content_array[i])
        elif expected_array[i] == 'Film':
            film = film + ' ' + str(content_array[i])
        elif expected_array[i] == 'Football':
            football = football + ' ' + str(content_array[i])
        elif expected_array[i] == 'Business':
            business = business + ' ' + str(content_array[i])
        elif expected_array[i] == 'Technology':
            technology = technology + ' ' + str(content_array[i])
            
    wordcloud = WordCloud(stopwords=STOPWORDS).generate(politics)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    
    wordcloud = WordCloud(stopwords=STOPWORDS).generate(film)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    
    wordcloud = WordCloud(stopwords=STOPWORDS).generate(football)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    
    wordcloud = WordCloud(stopwords=STOPWORDS).generate(business)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    
    wordcloud = WordCloud(stopwords=STOPWORDS).generate(technology)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

def clustering():
    clust_data = train_data[['Id', 'Title', 'Content']]
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(clust_data.Title)
    X_train_tfidf = vectorizer.transform(clust_data.Title)
    svd = TruncatedSVD(n_components = 100)
    X_lsi = svd.fit_transform(X_train_tfidf)

    clusterer = KMeansClusterer(5, cosine_distance)
    clusters = clusterer.cluster(X_lsi, True)
    
    clusters_data = np.zeros((5,5), dtype=float)
    c = np.zeros((5), dtype=int)
    
    for i in range(len(clusters)):
        if expected_array[i] == 'Politics':
            clusters_data[clusters[i]][0] += 1.0
            c[clusters[i]] += 1
        elif expected_array[i] == 'Film':
            clusters_data[clusters[i]][1] += 1.0
            c[clusters[i]] += 1
        elif expected_array[i] == 'Football':
            clusters_data[clusters[i]][2] += 1.0
            c[clusters[i]] += 1
        elif expected_array[i] == 'Business':
            clusters_data[clusters[i]][3] += 1.0
            c[clusters[i]] += 1
        elif expected_array[i] == 'Technology':
            clusters_data[clusters[i]][4] += 1.0
            c[clusters[i]] += 1
             
    for i in range(5):
        clusters_data[0][i] = round(clusters_data[0][i]/c[0], 2)
        clusters_data[1][i] = round(clusters_data[1][i]/c[1], 2)
        clusters_data[2][i] = round(clusters_data[2][i]/c[2], 2)
        clusters_data[3][i] = round(clusters_data[3][i]/c[3], 2)
        clusters_data[4][i] = round(clusters_data[4][i]/c[4], 2)
        
    rows = pd.Index(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'])
    columns = pd.Index(['Politics', 'Film', 'Football', 'Business', 'Technology'])
    output_df = pd.DataFrame(clusters_data, index=rows, columns=columns)
    output_df.to_csv('clustering_KMeans.csv', sep='\t')
    
def classification():
    rf = RandomForestClassifier()
    sv = svm.SVC(kernel='linear', C = 1.0)
    nb = MultinomialNB()

    count_vect = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
    count_vect.fit(train_data.Title)

    kf = KFold(n_splits=10)
    fold = 0
    for train_index, test_index in kf.split(train_data):
        train_vect = count_vect.transform(np.array(train_data.Title)[train_index])
        test_vect = count_vect.transform(np.array(train_data.Title)[test_index])
            
        rf.fit(train_vect, expected_array[train_index])
        sv.fit(train_vect, expected_array[train_index])
        nb.fit(train_vect, expected_array[train_index])
        
        rf_pred = rf.predict(test_vect)
        sv_pred = sv.predict(test_vect)
        nb_pred = nb.predict(test_vect)
            
        fold += 1
        print("Fold " + str(fold))
        print("Random Forests report")
        print(classification_report(rf_pred, expected_array[test_index], target_names=['Politics','Film','Football','Business','Technology']))
        print("SVM report")
        print(classification_report(sv_pred, expected_array[test_index], target_names=['Politics','Film','Football','Business','Technology']))
        print("Naive Bayes report")
        print(classification_report(nb_pred, expected_array[test_index], target_names=['Politics','Film','Football','Business','Technology']))

    #accuracy = cross_val_score(rf, train_vect, expected_array, cv=kf)
    #precision = cross_val_score(rf, train_vect, expected_array, cv=kf, scoring='precision_weighted')
    #recall = cross_val_score(rf, train_vect, expected_array, cv=kf, scoring='recall_weighted')
    #print(accuracy.mean() + " " + precision.mean() + " " + recall.mean())

    train_vect = count_vect.transform(train_data.Title)
    rf.fit(train_vect, expected_array)
    sv.fit(train_vect, expected_array)
    nb.fit(train_vect, expected_array)
    test_vect = count_vect.transform(test_data.Title)
    predicted_rf = rf.predict(test_vect)
    predicted_sv = sv.predict(test_vect)
    predicted_nb = nb.predict(test_vect)

    to_print_arr = np.array(test_data[['Id']]).flatten()
    for i in range(test_vect.shape[0]):
        print('%s %s => rf %s sv %s nb %s' % (str(i),to_print_arr[i], predicted_rf[i], predicted_sv[i], predicted_nb[i]))
        
    columns = pd.Index(['ID', 'Predicted_Category'])
    data = np.column_stack((to_print_arr,predicted_nb))
    output_df = pd.DataFrame(data, index=None, columns=columns)
    output_df.to_csv('testSet_categories.csv', sep='\t')
        
def main():
    word_cloud()
    clustering()
    classification()
    
if __name__ == "__main__":
    main()
