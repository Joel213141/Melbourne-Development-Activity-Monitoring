{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Importing libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 302,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import sklearn\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.datasets import make_blobs\n",
    "from yellowbrick.cluster import KElbowVisualizer\n",
    "from sklearn import metrics\n",
    "from sklearn.cluster import DBSCAN\n",
    "from matplotlib import pyplot as plt\n",
    "from scipy.cluster.hierarchy import dendrogram\n",
    "from sklearn.cluster import AgglomerativeClustering\n",
    "from sklearn.impute import SimpleImputer\n",
    "from scipy.spatial.distance import euclidean, cosine"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Data Analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Load the data from supplied data file. Print the data dimension"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 303,
   "metadata": {},
   "outputs": [],
   "source": [
    "data=pd.read_csv(r\"C:\\Users\\User\\Desktop\\Class Folder\\720\\development-activity-monitor.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 304,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1424, 42)"
      ]
     },
     "execution_count": 304,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Display the data type of all features. If the data type is integer, print the median values of the features. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 305,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "data_format                     object\n",
       "development_key                 object\n",
       "status                          object\n",
       "year_completed                 float64\n",
       "clue_small_area                 object\n",
       "clue_block                       int64\n",
       "street_address                  object\n",
       "property_id                      int64\n",
       "property_id_2                  float64\n",
       "property_id_3                  float64\n",
       "property_id_4                  float64\n",
       "property_id_5                  float64\n",
       "floors_above                     int64\n",
       "resi_dwellings                   int64\n",
       "studio_dwe                       int64\n",
       "one_bdrm_dwe                     int64\n",
       "two_bdrm_dwe                     int64\n",
       "three_bdrm_dwe                   int64\n",
       "student_apartments               int64\n",
       "student_beds                     int64\n",
       "student_accommodation_units      int64\n",
       "institutional_accom_beds         int64\n",
       "hotel_rooms                      int64\n",
       "serviced_apartments              int64\n",
       "hotels_serviced_apartments       int64\n",
       "hostel_beds                      int64\n",
       "childcare_places                 int64\n",
       "office_flr                       int64\n",
       "retail_flr                       int64\n",
       "industrial_flr                   int64\n",
       "storage_flr                      int64\n",
       "education_flr                    int64\n",
       "hospital_flr                     int64\n",
       "recreation_flr                   int64\n",
       "publicdispaly_flr                int64\n",
       "community_flr                    int64\n",
       "car_spaces                       int64\n",
       "bike_spaces                      int64\n",
       "town_planning_application       object\n",
       "longitude                      float64\n",
       "latitude                       float64\n",
       "geopoint                        object\n",
       "dtype: object"
      ]
     },
     "execution_count": 305,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 306,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "median of clue_block is 432.0 \n",
      "median of property_id is 109147.0 \n",
      "median of floors_above is 8.0 \n",
      "median of resi_dwellings is 4.0 \n",
      "median of studio_dwe is 0.0 \n",
      "median of one_bdrm_dwe is 0.0 \n",
      "median of two_bdrm_dwe is 0.0 \n",
      "median of three_bdrm_dwe is 0.0 \n",
      "median of student_apartments is 0.0 \n",
      "median of student_beds is 0.0 \n",
      "median of student_accommodation_units is 0.0 \n",
      "median of institutional_accom_beds is 0.0 \n",
      "median of hotel_rooms is 0.0 \n",
      "median of serviced_apartments is 0.0 \n",
      "median of hotels_serviced_apartments is 0.0 \n",
      "median of hostel_beds is 0.0 \n",
      "median of childcare_places is 0.0 \n",
      "median of office_flr is 0.0 \n",
      "median of retail_flr is 0.0 \n",
      "median of industrial_flr is 0.0 \n",
      "median of storage_flr is 0.0 \n",
      "median of education_flr is 0.0 \n",
      "median of hospital_flr is 0.0 \n",
      "median of recreation_flr is 0.0 \n",
      "median of publicdispaly_flr is 0.0 \n",
      "median of community_flr is 0.0 \n",
      "median of car_spaces is 0.0 \n",
      "median of bike_spaces is 0.0 \n"
     ]
    }
   ],
   "source": [
    "for i in range(data.shape[1]):\n",
    "    if ((data.iloc[:,i]).dtype == 'int64'):\n",
    "        print(f\"median of {data.columns[i]} is {data.iloc[:,i].median()} \")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "####  Print all the possible values of the feature “status” and calculate the ratio of each “status” value. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 307,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>data_format</th>\n",
       "      <th>development_key</th>\n",
       "      <th>status</th>\n",
       "      <th>year_completed</th>\n",
       "      <th>clue_small_area</th>\n",
       "      <th>clue_block</th>\n",
       "      <th>street_address</th>\n",
       "      <th>property_id</th>\n",
       "      <th>property_id_2</th>\n",
       "      <th>property_id_3</th>\n",
       "      <th>...</th>\n",
       "      <th>hospital_flr</th>\n",
       "      <th>recreation_flr</th>\n",
       "      <th>publicdispaly_flr</th>\n",
       "      <th>community_flr</th>\n",
       "      <th>car_spaces</th>\n",
       "      <th>bike_spaces</th>\n",
       "      <th>town_planning_application</th>\n",
       "      <th>longitude</th>\n",
       "      <th>latitude</th>\n",
       "      <th>geopoint</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Pre May 16</td>\n",
       "      <td>X000479</td>\n",
       "      <td>COMPLETED</td>\n",
       "      <td>2006.0</td>\n",
       "      <td>North Melbourne</td>\n",
       "      <td>342</td>\n",
       "      <td>191-201 Abbotsford Street NORTH MELBOURNE VIC ...</td>\n",
       "      <td>100023</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>144.945030</td>\n",
       "      <td>-37.802822</td>\n",
       "      <td>-37.80282184, 144.9450298</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Pre May 16</td>\n",
       "      <td>X000459</td>\n",
       "      <td>COMPLETED</td>\n",
       "      <td>2005.0</td>\n",
       "      <td>North Melbourne</td>\n",
       "      <td>333</td>\n",
       "      <td>218-224 Abbotsford Street NORTH MELBOURNE VIC ...</td>\n",
       "      <td>100119</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>144.945948</td>\n",
       "      <td>-37.802049</td>\n",
       "      <td>-37.80204879, 144.9459475</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Pre May 16</td>\n",
       "      <td>X000573</td>\n",
       "      <td>COMPLETED</td>\n",
       "      <td>2013.0</td>\n",
       "      <td>West Melbourne (Residential)</td>\n",
       "      <td>414</td>\n",
       "      <td>56-62 Abbotsford Street WEST MELBOURNE VIC 3003</td>\n",
       "      <td>100144</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>144.944719</td>\n",
       "      <td>-37.806791</td>\n",
       "      <td>-37.80679128, 144.9447186</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Pre May 16</td>\n",
       "      <td>X000563</td>\n",
       "      <td>COMPLETED</td>\n",
       "      <td>2014.0</td>\n",
       "      <td>West Melbourne (Residential)</td>\n",
       "      <td>409</td>\n",
       "      <td>1-9 Stawell Street WEST MELBOURNE VIC 3003</td>\n",
       "      <td>100441</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>28</td>\n",
       "      <td>0</td>\n",
       "      <td>144.942096</td>\n",
       "      <td>-37.806072</td>\n",
       "      <td>-37.80607242, 144.9420962</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Pre May 16</td>\n",
       "      <td>X000997</td>\n",
       "      <td>COMPLETED</td>\n",
       "      <td>2007.0</td>\n",
       "      <td>North Melbourne</td>\n",
       "      <td>1012</td>\n",
       "      <td>229-235 Arden Street NORTH MELBOURNE VIC 3051</td>\n",
       "      <td>100556</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>144.939286</td>\n",
       "      <td>-37.800374</td>\n",
       "      <td>-37.80037382, 144.9392856</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 42 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "  data_format development_key     status  year_completed  \\\n",
       "0  Pre May 16         X000479  COMPLETED          2006.0   \n",
       "1  Pre May 16         X000459  COMPLETED          2005.0   \n",
       "2  Pre May 16         X000573  COMPLETED          2013.0   \n",
       "3  Pre May 16         X000563  COMPLETED          2014.0   \n",
       "4  Pre May 16         X000997  COMPLETED          2007.0   \n",
       "\n",
       "                clue_small_area  clue_block  \\\n",
       "0               North Melbourne         342   \n",
       "1               North Melbourne         333   \n",
       "2  West Melbourne (Residential)         414   \n",
       "3  West Melbourne (Residential)         409   \n",
       "4               North Melbourne        1012   \n",
       "\n",
       "                                      street_address  property_id  \\\n",
       "0  191-201 Abbotsford Street NORTH MELBOURNE VIC ...       100023   \n",
       "1  218-224 Abbotsford Street NORTH MELBOURNE VIC ...       100119   \n",
       "2    56-62 Abbotsford Street WEST MELBOURNE VIC 3003       100144   \n",
       "3         1-9 Stawell Street WEST MELBOURNE VIC 3003       100441   \n",
       "4      229-235 Arden Street NORTH MELBOURNE VIC 3051       100556   \n",
       "\n",
       "   property_id_2  property_id_3  ...  hospital_flr  recreation_flr  \\\n",
       "0            NaN            NaN  ...             0               0   \n",
       "1            NaN            NaN  ...             0               0   \n",
       "2            NaN            NaN  ...             0               0   \n",
       "3            NaN            NaN  ...             0               0   \n",
       "4            NaN            NaN  ...             0               0   \n",
       "\n",
       "   publicdispaly_flr  community_flr  car_spaces  bike_spaces  \\\n",
       "0                  0              0           0            0   \n",
       "1                  0              0           0            0   \n",
       "2                  0              0           0            0   \n",
       "3                  0              0           0           28   \n",
       "4                  0              0           0            0   \n",
       "\n",
       "   town_planning_application   longitude   latitude                   geopoint  \n",
       "0                          0  144.945030 -37.802822  -37.80282184, 144.9450298  \n",
       "1                          0  144.945948 -37.802049  -37.80204879, 144.9459475  \n",
       "2                          0  144.944719 -37.806791  -37.80679128, 144.9447186  \n",
       "3                          0  144.942096 -37.806072  -37.80607242, 144.9420962  \n",
       "4                          0  144.939286 -37.800374  -37.80037382, 144.9392856  \n",
       "\n",
       "[5 rows x 42 columns]"
      ]
     },
     "execution_count": 307,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 308,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1424, 42)"
      ]
     },
     "execution_count": 308,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 309,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['APPLIED', 'APPROVED', 'COMPLETED', 'UNDER CONSTRUCTION'],\n",
       "      dtype=object)"
      ]
     },
     "execution_count": 309,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "uni_status=np.unique(data[\"status\"])\n",
    "uni_status"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 310,
   "metadata": {},
   "outputs": [],
   "source": [
    "values=data['status'].value_counts()\n",
    "sorted_values= values.sort_index()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 311,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The ratio of APPLIED is 0.06671348314606741\n",
      "The ratio of APPROVED is 0.1720505617977528\n",
      "The ratio of COMPLETED is 0.6973314606741573\n",
      "The ratio of UNDER CONSTRUCTION is 0.06390449438202248\n"
     ]
    }
   ],
   "source": [
    "uni=np.unique(data[\"status\"])\n",
    "for i in range (len(uni)):\n",
    "    print(f\"The ratio of {uni[i]} is {sorted_values[i]/sum(values)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 312,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0                COMPLETED\n",
       "1                COMPLETED\n",
       "2                COMPLETED\n",
       "3                COMPLETED\n",
       "4                COMPLETED\n",
       "               ...        \n",
       "1419    UNDER CONSTRUCTION\n",
       "1420    UNDER CONSTRUCTION\n",
       "1421    UNDER CONSTRUCTION\n",
       "1422    UNDER CONSTRUCTION\n",
       "1423    UNDER CONSTRUCTION\n",
       "Name: status, Length: 1424, dtype: object"
      ]
     },
     "execution_count": 312,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['status']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 313,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0       100023\n",
       "1       100119\n",
       "2       100144\n",
       "3       100441\n",
       "4       100556\n",
       "         ...  \n",
       "1419    636830\n",
       "1420    108056\n",
       "1421    105720\n",
       "1422    105631\n",
       "1423    102725\n",
       "Name: property_id, Length: 1424, dtype: int64"
      ]
     },
     "execution_count": 313,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['property_id']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 314,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.023280182023912004"
      ]
     },
     "execution_count": 314,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import  normalized_mutual_info_score\n",
    "normalized_mutual_info_score(data['status'], data['clue_small_area'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Association between 'clue small area' and 'status':\n",
    "* Under clue small area, the areas that have the largest number of properties that have completed. \n",
    "* From the mutual information score calculated between the two features, I can conclude that the relationship between status and clue_small_area is quite weak even for a highly dimensional dataset such as this one. \n",
    "* A theoretical association between the two could determine under which area most proterties have the majority of the respective statuses with respect to the number of properties in the area."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Number of properties for different suburbs."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 315,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Melbourne (CBD)                 334\n",
       "Docklands                       198\n",
       "North Melbourne                 186\n",
       "Carlton                         137\n",
       "Southbank                       133\n",
       "West Melbourne (Residential)    112\n",
       "Kensington                       85\n",
       "Port Melbourne                   78\n",
       "Parkville                        61\n",
       "East Melbourne                   42\n",
       "West Melbourne (Industrial)      21\n",
       "Melbourne (Remainder)            21\n",
       "South Yarra                      16\n",
       "Name: clue_small_area, dtype: int64"
      ]
     },
     "execution_count": 315,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['clue_small_area'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Patterns Found: \n",
    "* The areas surrounding the melbourne CBD such as Docklands, North Melbourne, Southbank, Carlton and the CBD itself contain the most number of properties. \n",
    "* The residentital part of west melbourne contains a high number of properties as people live there compared to the industrial area with fewer properties. \n",
    "* The areas furthurest away from the city contains the least amount of properties such as Parkville and South Yarra. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### which suburb has the biggest number of properties which are under construction?\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 316,
   "metadata": {},
   "outputs": [],
   "source": [
    "uni_subs=np.unique(data['clue_small_area'])\n",
    "under=[]\n",
    "for i in range (len(data)):\n",
    "    if (data[\"status\"][i]=='UNDER CONSTRUCTION'):\n",
    "        under.append(data['clue_small_area'][i])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 317,
   "metadata": {},
   "outputs": [],
   "source": [
    "under_dataframe=pd.DataFrame(under)\n",
    "unique, counts = np.unique(under_dataframe, return_counts=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 318,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Carlton', 'Docklands', 'East Melbourne', ..., 'Southbank',\n",
       "       'West Melbourne (Industrial)', 'West Melbourne (Residential)'],\n",
       "      dtype=object)"
      ]
     },
     "execution_count": 318,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "unique"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 319,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 4,  9,  5, ...,  5,  1, 11], dtype=int64)"
      ]
     },
     "execution_count": 319,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "counts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 320,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Melbourne (CBD)'"
      ]
     },
     "execution_count": 320,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "unique[np.argmax(counts)]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### which suburb has the biggest number of student apartments?\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 321,
   "metadata": {},
   "outputs": [],
   "source": [
    "uni_subs=np.unique(data['clue_small_area'])\n",
    "stud=[]\n",
    "summ=0\n",
    "for i in range (len(uni_subs)):\n",
    "    summ=0\n",
    "    for j in range (len(data)):\n",
    "        if (uni_subs[i]==data['clue_small_area'][j]):\n",
    "            summ=summ+data['student_apartments'][j]\n",
    "    stud.append(summ)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 322,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[7510, 0, 0, 0, 7283, 0, 1506, 606, 0, 0, 0, 0, 321]"
      ]
     },
     "execution_count": 322,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "stud"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 323,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Carlton', 'Docklands', 'East Melbourne', ..., 'Southbank',\n",
       "       'West Melbourne (Industrial)', 'West Melbourne (Residential)'],\n",
       "      dtype=object)"
      ]
     },
     "execution_count": 323,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "uni_subs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 324,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Carlton'"
      ]
     },
     "execution_count": 324,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "uni_subs[np.argmax(stud)]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Create and print a data frame of the number of different status values for different year groups (based on 5 years interval).\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 325,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2002., 2003., 2004., ..., 2021., 2022.,   nan])"
      ]
     },
     "execution_count": 325,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.unique(data['year_completed'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 326,
   "metadata": {},
   "outputs": [],
   "source": [
    "year_groups=[\"2000-2005\",\"2006-2010\",\"2011-2015\",\"2016-2020\",\"2021-2025\"]\n",
    "completed_nums=[]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 327,
   "metadata": {},
   "outputs": [],
   "source": [
    "summ=[0,0,0,0,0]\n",
    "for j in range(len(data)):\n",
    "    if (((data['year_completed'][j])>2000.0) & ((data['year_completed'][j])<2006.0)):\n",
    "        summ[0]=summ[0]+1\n",
    "    elif (((data['year_completed'][j])>2005.0) & ((data['year_completed'][j])<2011.0)):\n",
    "        summ[1]=summ[1]+1\n",
    "    elif (((data['year_completed'][j])>2010.0) & ((data['year_completed'][j])<2016.0)):\n",
    "        summ[2]=summ[2]+1\n",
    "    elif (((data['year_completed'][j])>2015.0) & ((data['year_completed'][j])<2021.0)):\n",
    "        summ[3]=summ[3]+1\n",
    "    elif (((data['year_completed'][j])>2020.0) & ((data['year_completed'][j])<2026.0)):\n",
    "        summ[4]=summ[4]+1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 328,
   "metadata": {},
   "outputs": [],
   "source": [
    "summ\n",
    "col={\"Year_Groups\" : year_groups,\"Completed_Total\" : summ }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 329,
   "metadata": {},
   "outputs": [],
   "source": [
    "new_data= pd.DataFrame(col)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 330,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Year_Groups</th>\n",
       "      <th>Completed_Total</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2000-2005</td>\n",
       "      <td>149</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2006-2010</td>\n",
       "      <td>243</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2011-2015</td>\n",
       "      <td>292</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2016-2020</td>\n",
       "      <td>241</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2021-2025</td>\n",
       "      <td>68</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Year_Groups  Completed_Total\n",
       "0   2000-2005              149\n",
       "1   2006-2010              243\n",
       "2   2011-2015              292\n",
       "3   2016-2020              241\n",
       "4   2021-2025               68"
      ]
     },
     "execution_count": 330,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new_data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### A histogram of number of status values against different year groups.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 331,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<BarContainer object of 5 artists>"
      ]
     },
     "execution_count": 331,
     "metadata": {},
     "output_type": "execute_result"
    },
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
    "plt.bar(year_groups, summ)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Alanysis: \n",
    "* From the Histogram, it can be clearly noted that from the year 2011-2015, most of the properties were completed. \n",
    "* There has been an overall decrease in the completion of properties from the year group 2000-2005 to 2021-2025"
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
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### perform K-Means and Hierarchical clustering on the data set, report the purity score. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### K-Means"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 337,
   "metadata": {},
   "outputs": [],
   "source": [
    "clusterer = KMeans(n_clusters=10, random_state=0)\n",
    "cluster_labels = clusterer.fit_predict(imputed_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 338,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "purity_score: 0.247\n"
     ]
    }
   ],
   "source": [
    "from sklearn import metrics\n",
    "\n",
    "def purity_score(y_true, y_pred):\n",
    "    # compute contingency matrix (also called confusion matrix)\n",
    "    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)\n",
    "    # return purity\n",
    "    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) \n",
    "print(\n",
    "    \"purity_score: %0.3f\"\n",
    "    % purity_score(truth, cluster_labels)\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 339,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2, ..., 7, 8, 9])"
      ]
     },
     "execution_count": 339,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.unique(cluster_labels)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Hierarchial Clustering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 340,
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_dendrogram(model, **kwargs):\n",
    "    # Create linkage matrix and then plot the dendrogram\n",
    "\n",
    "    # create the counts of samples under each node\n",
    "    counts = np.zeros(model.children_.shape[0])\n",
    "    n_samples = len(model.labels_)\n",
    "    for i, merge in enumerate(model.children_):\n",
    "        current_count = 0\n",
    "        for child_idx in merge:\n",
    "            if child_idx < n_samples:\n",
    "                current_count += 1  # leaf node\n",
    "            else:\n",
    "                current_count += counts[child_idx - n_samples]\n",
    "        counts[i] = current_count\n",
    "\n",
    "    linkage_matrix = np.column_stack(\n",
    "        [model.children_, model.distances_, counts]\n",
    "    ).astype(float)\n",
    "\n",
    "    # Plot the corresponding dendrogram\n",
    "    dendrogram(linkage_matrix, **kwargs)"
   ]
  },
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
  },
  {
   "cell_type": "code",
   "execution_count": 343,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, ..., 0, 0, 0], dtype=int64)"
      ]
     },
     "execution_count": 343,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.labels_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 344,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "purity_score: 0.247\n"
     ]
    }
   ],
   "source": [
    "print(\n",
    "    \"purity_score: %0.3f\"\n",
    "    % purity_score(truth, model.labels_)\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 345,
   "metadata": {},
   "outputs": [],
   "source": [
    "np.set_printoptions(threshold=3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 12. Three different distance metrics for K-Means and Hierarchical clustering, select the best distance metric for each corresponding clustering algorithm. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For K means"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 346,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer\n",
    "intializer=kmeans_plusplus_initializer(imputed_data,10).initialize()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "MANHATTEN Distance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 347,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pyclustering\n",
    "from pyclustering.utils.metric import distance_metric, type_metric\n",
    "from pyclustering.cluster.encoder import cluster_encoder\n",
    "from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer\n",
    "manhattan_metric = distance_metric(type_metric.MANHATTAN)\n",
    "kmeans_instance = kmeans(imputed_data, initial_centers=intializer, metric=manhattan_metric)\n",
    "kmeans_instance.process()\n",
    "clusters = kmeans_instance.get_clusters()\n",
    "pyEncoding = kmeans_instance.get_cluster_encoding()\n",
    "pyEncoder = cluster_encoder(pyEncoding, clusters, imputed_data)\n",
    "# change representation from index list to label list\n",
    "pyLabels = pyEncoder.set_encoding(pyclustering.cluster.encoder.type_encoding.CLUSTER_INDEX_LABELING.value).get_clusters()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 348,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2, ..., 7, 8, 9])"
      ]
     },
     "execution_count": 348,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.unique(pyLabels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 349,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "purity_score: 0.247\n"
     ]
    }
   ],
   "source": [
    "print(\n",
    "    \"purity_score: %0.3f\"\n",
    "    % purity_score(truth, pyLabels)\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "EUCLIDEAN distance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 354,
   "metadata": {},
   "outputs": [],
   "source": [
    "manhattan_metric = distance_metric(type_metric.EUCLIDEAN)\n",
    "kmeans_instance = kmeans(imputed_data, initial_centers=intializer, metric=manhattan_metric)\n",
    "kmeans_instance.process()\n",
    "clusters = kmeans_instance.get_clusters()\n",
    "pyEncoding = kmeans_instance.get_cluster_encoding()\n",
    "pyEncoder = cluster_encoder(pyEncoding, clusters, imputed_data)\n",
    "# change representation from index list to label list\n",
    "pyLabels = pyEncoder.set_encoding(pyclustering.cluster.encoder.type_encoding.CLUSTER_INDEX_LABELING.value).get_clusters()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 356,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "purity_score: 0.247\n"
     ]
    }
   ],
   "source": [
    "print(\n",
    "    \"purity_score: %0.3f\"\n",
    "    % purity_score(truth, pyLabels)\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "CHEBYSHEV Distance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 357,
   "metadata": {},
   "outputs": [],
   "source": [
    "manhattan_metric = distance_metric(type_metric.CHEBYSHEV)\n",
    "kmeans_instance = kmeans(imputed_data, initial_centers=intializer, metric=manhattan_metric)\n",
    "kmeans_instance.process()\n",
    "clusters = kmeans_instance.get_clusters()\n",
    "pyEncoding = kmeans_instance.get_cluster_encoding()\n",
    "pyEncoder = cluster_encoder(pyEncoding, clusters, imputed_data)\n",
    "# change representation from index list to label list\n",
    "pyLabels = pyEncoder.set_encoding(pyclustering.cluster.encoder.type_encoding.CLUSTER_INDEX_LABELING.value).get_clusters()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 358,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "purity_score: 0.247\n"
     ]
    }
   ],
   "source": [
    "print(\n",
    "    \"purity_score: %0.3f\"\n",
    "    % purity_score(truth, pyLabels)\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "From reviewing the above purity score, we can conlcude that all the distance metrics are equally viable as they all have the same purity score. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Heirarchial Clustering"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "MANHATTEN Distance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 359,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = AgglomerativeClustering(compute_distances=True,linkage=\"complete\", affinity=\"manhattan\",n_clusters=10)\n",
    "model = model.fit(imputed_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 360,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "purity_score: 0.248\n"
     ]
    }
   ],
   "source": [
    "print(\n",
    "    \"purity_score: %0.3f\"\n",
    "    % purity_score(truth, model.labels_)\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "EUCLIDEAN Distance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 361,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "purity_score: 0.239\n"
     ]
    }
   ],
   "source": [
    "model = AgglomerativeClustering(compute_distances=True,linkage=\"complete\", affinity=\"euclidean\",n_clusters=2)\n",
    "model = model.fit(imputed_data)\n",
    "print(\n",
    "    \"purity_score: %0.3f\"\n",
    "    % purity_score(truth, model.labels_)\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "CHEBYSHEV Distance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 362,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "purity_score: 0.240\n"
     ]
    }
   ],
   "source": [
    "model = AgglomerativeClustering(compute_distances=True,linkage=\"complete\", affinity=\"chebyshev\",n_clusters=2)\n",
    "model = model.fit(imputed_data)\n",
    "print(\n",
    "    \"purity_score: %0.3f\"\n",
    "    % purity_score(truth, model.labels_)\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "From reviewing the above purity scores for heirarchial clustering we can conclude that Manhattan distance is the best distance metrics for this dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### DBSCAN clustering method, and compare the results. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 363,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Optimal eps: 30; Optimal min_samples: 6; optimal number of cluster: 2\n",
      "the max silhouette score: 0.5789290319835195\n",
      "the ground truth of labels : 13 the labels_pre:  2\n",
      "purity_score: 0.239\n"
     ]
    }
   ],
   "source": [
    "from sklearn.cluster import DBSCAN\n",
    "silhouette_scores = []\n",
    "for eps in range(10, 35):\n",
    "    for min_samples in range(5, 25):\n",
    "        dbscan = DBSCAN(eps=eps, min_samples=min_samples)\n",
    "        dbscan.fit(imputed_data)\n",
    "        labelsData = dbscan.labels_\n",
    "        unique_label_num = len(set(labelsData))\n",
    "        if unique_label_num > 1 & unique_label_num < len(imputed_data):\n",
    "            try:\n",
    "                score = silhouette_score(imputed_data, labelsData)\n",
    "            except ValueError:\n",
    "                raise\n",
    "\n",
    "            silhouette_scores.append((score, eps, min_samples, unique_label_num,labelsData))\n",
    "\n",
    "max_silhouette_score, optimal_eps, optimal_min_samples, optimal_n_label, labels_pre = max(silhouette_scores)[:]\n",
    "print(f\"Optimal eps: {optimal_eps}; Optimal min_samples: {optimal_min_samples}; optimal number of cluster: {optimal_n_label}\")\n",
    "print(f\"the max silhouette score: {max_silhouette_score}\")\n",
    "print(\"the ground truth of labels :\", len(set(truth)), \"the labels_pre: \", len(set(labels_pre)))\n",
    "print(\n",
    "    \"purity_score: %0.3f\"\n",
    "    % purity_score(truth, labels_pre)\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here the max silhouette score is very low compared to the ones obtained for K-means and Heirarchial clustering, but the purity scores appears to be lower but close. From this we can conclude that DBSCAN is not a good clustering method for this dataset. "
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
