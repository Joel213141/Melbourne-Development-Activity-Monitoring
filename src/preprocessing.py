{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Data preprocessing"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Based on the original dataset, exclude the clue_small_area feature, use the rest available features and perform clustering on all the properties and determine the number of clusters.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 332,
   "metadata": {},
   "outputs": [],
   "source": [
    "n_data=data.drop('clue_small_area',axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Label Encoding the string data in the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 333,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "label_encoder= LabelEncoder()\n",
    "n_data[\"development_key\"] = label_encoder.fit_transform(n_data['development_key'])\n",
    "n_data[\"status\"] = label_encoder.fit_transform(n_data['status'])\n",
    "n_data[\"town_planning_application\"] = label_encoder.fit_transform(n_data['town_planning_application'])\n",
    "n_data[\"data_format\"] = label_encoder.fit_transform(n_data['data_format'])\n",
    "n_data[\"street_address\"] = label_encoder.fit_transform(n_data['street_address'])\n",
    "n_data[\"geopoint\"] = label_encoder.fit_transform(n_data['geopoint'])\n",
    "n_data[\"clue_block\"] = label_encoder.fit_transform(n_data['clue_block']) # unique identifier\n",
    "n_data[\"property_id\"] = label_encoder.fit_transform(n_data['property_id']) # unique identifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 334,
   "metadata": {},
   "outputs": [],
   "source": [
    "imp = SimpleImputer(missing_values=np.nan, strategy='mean')\n",
    "imputed_data=imp.fit_transform(n_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 335,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "For n_clusters = 2 The average silhouette_score is : 0.9218744457771927\n",
      "For n_clusters = 3 The average silhouette_score is : 0.893286982709299\n",
      "For n_clusters = 4 The average silhouette_score is : 0.9012308420229768\n",
      "For n_clusters = 5 The average silhouette_score is : 0.9109259230745754\n",
      "For n_clusters = 6 The average silhouette_score is : 0.9254764498256866\n",
      "For n_clusters = 7 The average silhouette_score is : 0.9277950172224734\n",
      "For n_clusters = 8 The average silhouette_score is : 0.932112462571412\n",
      "For n_clusters = 9 The average silhouette_score is : 0.9323245175537551\n",
      "For n_clusters = 10 The average silhouette_score is : 0.9331385009218566\n",
      "For n_clusters = 11 The average silhouette_score is : 0.8768829172427629\n",
      "For n_clusters = 12 The average silhouette_score is : 0.8715724545327229\n",
      "For n_clusters = 13 The average silhouette_score is : 0.871893480987058\n",
      "For n_clusters = 14 The average silhouette_score is : 0.8717448460703169\n",
      "For n_clusters = 15 The average silhouette_score is : 0.8759535266676804\n"
     ]
    }
   ],
   "source": [
    "# Selecting the optimum k value using Silhouette Coefficient and plot the optimum k values. \n",
    "from sklearn.datasets import make_blobs\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.metrics import silhouette_samples, silhouette_score\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.cm as cm\n",
    "import numpy as np\n",
    "silouette_values=[]\n",
    "range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13,14,15]\n",
    "\n",
    "for n_clusters in range_n_clusters:\n",
    "    # Initialize the clusterer with n_clusters value and a random generator\n",
    "    # seed of 10 for reproducibility.\n",
    "    clusterer = KMeans(n_clusters=n_clusters, random_state=10)\n",
    "    cluster_labels = clusterer.fit_predict(imputed_data)\n",
    "\n",
    "    # The silhouette_score gives the average value for all the samples.\n",
    "    # This gives a perspective into the density and separation of the formed\n",
    "    # clusters\n",
    "    silhouette_avg = silhouette_score(imputed_data, cluster_labels)\n",
    "    silouette_values.append(silhouette_score(imputed_data, cluster_labels))\n",
    "    print(\n",
    "        \"For n_clusters =\",\n",
    "        n_clusters,\n",
    "        \"The average silhouette_score is :\",\n",
    "        silhouette_avg,\n",
    "    )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    " The ideal number of clusters from above can be clearly inferred to be 10."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 336,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0,  1,  2, ..., 10, 11, 12])"
      ]
     },
     "execution_count": 336,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "label_encoder= LabelEncoder()\n",
    "truth = label_encoder.fit_transform(data['clue_small_area'])\n",
    "np.unique(truth)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "From above it can be inferred that there are 13 unique suburbs in the dataset and the ideal number of clusters come out to be 10. Therefore they are not the same. "
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
