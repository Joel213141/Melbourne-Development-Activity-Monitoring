{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 341,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = AgglomerativeClustering(compute_distances=True,linkage=\"complete\", affinity=\"cosine\",n_clusters=10)\n",
    "model = model.fit(imputed_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 342,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.title(\"Hierarchical Clustering Dendrogram\")\n",
    "# plot the top three levels of the dendrogram\n",
    "plot_dendrogram(model, truncate_mode=\"level\", p=3)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
