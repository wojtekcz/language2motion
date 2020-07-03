
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca_result = pca.fit_transform(X[:])

df = ann_df

df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
df['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# plt.figure(figsize=(30,20))
plt.figure(figsize=(15,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="pred",
    palette=sns.color_palette("hls", true_k),
    data=df, #df.loc[0,:],
#     data=tsne_df,
    legend="full",
    alpha=1.0 #0.3
)

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=df["pca-one"], 
    ys=df["pca-two"], 
    zs=df["pca-three"], 
    c=df["pred"], 
    cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()

