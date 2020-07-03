# # specify some clusters manually

# In[552]:


true_k = 5


# In[553]:


classes = ['person is running', 'person walks slowly', 'person turns right or left', 'person moves clockwise', 'person moves hand']
centroids_df = pd.DataFrame({'pred': list(range(true_k)), 'class': classes})
centers = bc.encode(classes)
centroids_df['center_init' ] = [centers[x] for x in range(centers.shape[0])]
centroids_df


# In[554]:


center_init_arr = np.stack(centroids_df.center_init.to_numpy(), axis=0)
center_init_arr.shape


# In[555]:


time_start = time.time()
# kmeans = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=10)
kmeans = KMeans(n_clusters=true_k, init=center_init_arr, max_iter=1, n_init=1)
# kmeans.fit(X)
# print('k-means done! Time elapsed: {} seconds'.format(time.time()-time_start))


# # compute distances

# In[557]:


v1 =center_init_arr[0:2]
v1.shape


# In[558]:


def assign_cluster(vec, centers):
#     dist = euclidean_distances(vec, centers)
    dist = cosine_similarity(vec, centers)
    print(dist)
#     return np.argmin(dist, axis=1)
    return np.argmax(dist, axis=1)
assign_cluster(v1, center_init_arr)


# In[ ]:





# In[559]:


# centroids_df = pd.DataFrame({'pred': list(range(true_k))})
# centers = kmeans.cluster_centers_
centers = center_init_arr
centroids_df['center'] = [centers[x] for x in range(centers.shape[0])]
centroids_df

clusters = kmeans.labels_.tolist()
len(clusters)Y_df = ann_df
Y = X
# In[560]:


len(X)


# In[561]:


# prediction = kmeans.predict(Y)
prediction = assign_cluster(X, center_init_arr)
print(prediction[:10])

ann_df['pred'] = prediction


# In[562]:


ann_df
