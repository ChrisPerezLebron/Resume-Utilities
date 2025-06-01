import gensim.downloader as api
path = api.load("word2vec-google-news-300", return_path=True)
print(path)