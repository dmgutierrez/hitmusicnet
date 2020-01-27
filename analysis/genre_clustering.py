# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 12:20:15 2018

@author: user
"""

import pandas as pd
import operator
import string
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import (dendrogram, linkage)
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
from helper.data_helper import read_dataframe, save_dataframe,get_popular_genres

#from sklearn.metrics.pairwise import cosine_similarity

def get_genre_df(df):
    df['Genre'] = df['Genre'].str.replace('[','').str.replace(']', '').str.replace('\'', '')
    
    # Clean Playlist Name
    df["Playlist"] = df.apply(lambda row:clean_playlist_name(row['Playlist']),
      axis=1)      
    return df

def get_genre_vocabulary(df,threshold=0):
    genres = Counter()
    for row in df["Genre"]:  
        if len(row)>0:
            g = row.split(', ')
            genres.update(g)
    # Dictionary with the principal genres of the playlist 
    vocabulary = Counter(dict(filter(lambda x: x[1] > threshold, genres.items())))   
    
    return vocabulary    


def create_Bag_of_Words_structure(df, vocabulary):
    # Dictionary with the songs and its genre
    songs = dict(zip(df["Name"], df["Genre"]))
    
    # Remove songs if there is no genre
    songs_cl = {k: v for k, v in songs.items() if v is not ''}
    
    # List of strings
    vals = list(songs_cl.values())
    
    genre_list_per_song = []
    for i in vals:
        genre_list_per_song.append(i.split(', '))
    
    # Create a dataframe with the BoW
    df_BoW = pd.DataFrame({'Name': list(songs_cl.keys()),
                           'Genres': genre_list_per_song})
    
    f = lambda x: Counter([y for y in x if y in vocabulary.keys()])
    
    df_BoW['BoW'] = (pd.DataFrame(df_BoW['Genres'].apply(f).values.tolist())
                   .fillna(0)
                   .astype(int)
                   .reindex(columns=vocabulary.keys())
                   .values
                   .tolist())        
    return df_BoW

def compute_similarity_matrix(df_BoW, vocabulary):
    list_values = list(df_BoW["BoW"])
    matrix = np.asmatrix(list_values)
    
    similarity_matrix = np.dot(matrix.T, matrix) #cosine_similarity(matrix)
    
    # squared magnitude of preference vectors (number of occurrences)
    square_mag = np.diag(similarity_matrix)
    
    # inverse squared magnitude
    inv_square_mag = 1 / square_mag
    
    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0
    
    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)
    
    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = inv_mag*similarity_matrix 
    distance_matrix = cosine.T * inv_mag
        
    #distance_matrix = 1 - similarity_matrix
    df_sim_matrix = pd.DataFrame(distance_matrix,
                                 columns=vocabulary.keys(),
                                 index=vocabulary.keys())
    return df_sim_matrix, distance_matrix


def plot_similarity_matrix(df_sim_matrix):
   
    # Plot confusion matrix
    plt.figure(figsize=(12,12))
    plt.imshow(df_sim_matrix,interpolation='none',cmap='RdPu')
    plt.xlabel("Genres")
    plt.ylabel("Genres")
    plt.colorbar()
    plt.title('Co-ocurrence matrix among the different genres of a playlist')
    plt.savefig('Figures/Similarity_matrix.png')
    plt.show()
    
def textRank(similarity_matrix,vocabulary):
    sparse_matrix = csr_matrix(similarity_matrix)
    nx_graph = nx.from_scipy_sparse_matrix(sparse_matrix)
    scores = nx.pagerank(nx_graph)
    
    sorted_scores = sorted(((scores[i],s) for i,s in enumerate(vocabulary.keys())),
                           reverse=True)
    return sorted_scores

def plot_dendrogram(df_sim_matrix):
    dendrogram_sns = sns.clustermap(df_sim_matrix,   cmap="Blues")
    dendrogram_sns.savefig('Figures/dendrogram_test3.png')
    
    # generate the linkage matrix
    Z = linkage(df_sim_matrix, metric='cosine', method='complete') 
    # calculate full dendrogram
    plt.figure(figsize=(16, 20))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('country')
    plt.ylabel('distance')
    dend = dendrogram(
        Z,
        leaf_font_size=10.,  # font size for the x axis labels
        labels = df_sim_matrix.index,
        orientation = 'left',
        distance_sort = True,
        show_contracted=True)
    plt.cm.get_cmap("Set3", 15)
    plt.savefig('Figures/semantic_dendrogram.png')
    plt.show()
    
    return dend,Z

def optimal_number_clusters(df_sim_matrix):
    cluster_range = range( 5, 30 )
    cluster_errors = []
    
    for num_clusters in cluster_range:
      clusters = KMeans( num_clusters )
      clusters.fit( df_sim_matrix )
      cluster_errors.append( clusters.inertia_ )
      
    clusters_df = pd.DataFrame( { "num_clusters":cluster_range,
                                 "cluster_errors": cluster_errors } )
    plt.figure(figsize=(12,6))
    plt.plot( clusters_df.num_clusters, 
             clusters_df.cluster_errors,
             marker = "o" )
    
    return True

def get_clusters(df, df_sim_matrix, vocabulary):
    k = 12
    model = AgglomerativeClustering(affinity='precomputed',
                                    n_clusters=k,
                                    linkage='complete').fit(df_sim_matrix)
    
    clusters = model.labels_
    df_cl = pd.DataFrame({'Genres': list(vocabulary), 'Clusters': clusters})
    return df_cl

def f(row, category="Clusters"):
    val=''
    if row[category]==1:
        val='Electronic'
    elif row[category]==2:
        val='Alternative Rock'
    elif row[category]==3:
        val='Alternative Pop'
    elif row[category]==4:
        val='Hip-Hop'
    elif row[category]==5:
        val='Latin'
    elif row[category]==6:
        val='Indie'
    elif row[category]==7:
        val='Electronic'
    elif row[category]==8:
        val='Pop'
    elif row[category]==9:
        val='Country'
    return val


def add_parent_genre(row):
    final_genre = ''
    current_genres = row['Genre'].replace('-', ' ').split(', ')
    
    general_genres = get_popular_genres()
    new_genres = dict(list(zip(general_genres.keys(),
                          np.zeros(len(general_genres.keys())))))
   
    for k,v in general_genres.items():
        # CASE 1: THERE ARE AVAILABLE GENRES IN THE ROW
        if len(current_genres)>=1 and '' not in current_genres:
            # Loop over current genres
            for g in current_genres:
                not_find = True
                #print(g)
                # If genre is in the general_genres sublist
                if g in  v:
                    #print('**********', g)
                    # Get the parent genre
                    new_genres[k] +=1
                    not_find = False # Find
                
                # Check using the parent if not find
                if not_find:
                    if 'hip hop' in g:
                        g = g.replace('hip hop', 'hip-hop')
                    split_current_genre = g.split(' ')
                    for subgen in split_current_genre:
                        # Just in case
                        subgen = subgen.replace('-', ' ')                 
                        if len(subgen)>2 and subgen in v:
                            # Get the parent genre
                            #print('---------- IN', subgen)
                            new_genres[k] +=1
                            not_find = False
                        if not_find and len(subgen)>2 and subgen in k.lower():
                            #print('---------- IN2', subgen)
                            new_genres[k] +=1
                            not_find = False           
                # Check in the Playlist Name (last chance)
                if not_find:
                    playlist_name = row['Playlist'].lower()
                    for subgg in v:
                        if subgg in playlist_name:
                            new_genres[k] +=1
                            not_find = False
        
        # CASE 2: THERE IS NO GENRE AVAILABLE
        else:
            # Check the playlist name
            playlist_name = row['Playlist'].lower()
            for subgg in v:
                not_find = True
                if subgg in playlist_name:
                    new_genres[k] +=1
                    not_find = False
                    
        # Get the values
        total_sum = sum(list(new_genres.values()))
        if total_sum>=1:
            # Get the parent with more genres associated
            final_genre = max(new_genres.items(), key=operator.itemgetter(1))[0]
    
    return final_genre
        

def compute_final_genre(df):
    new_genre =[]
    for i, row in df.iterrows():
         print(80*'=')
         print('......... Adding Genre {}/{} .........'.format(i+1, len(df)))
         print(80*'=')
         new_genre.append(add_parent_genre(row))
    df["General_genre"] = new_genre
    return df
    
def maximum_genre_replacement(df):
    # Get the most repetitive genre in the Playlist
    for playlist in list(pd.unique(df['Playlist'])):
        # Get the corresponding rows from each Playlist
        temp = df.loc[(df['Playlist']==playlist),]
        count_genre = dict(temp['General_genre'].value_counts())
        
        # Check number of nan's elements
        n_nan = len(temp.loc[(temp['General_genre']=='Unknown'),])
        nan_percentage = n_nan/len(temp)
        thres = 0.5
        
        if nan_percentage <= thres:
            new_genre_count = count_genre
            new_genre_count.pop('Unknown', None)
            # Get the maximum
            max_genre = max(new_genre_count.items(), key=operator.itemgetter(1))[0]
        else:
            max_genre = 'Unknown'
        
        # Replace the values
        df.loc[(df['Playlist']==playlist) & (df['General_genre']=='Unknown'),
               'General_genre'] = max_genre
  
        # Africa
        if 'Africa' in playlist:
            df.loc[(df['Playlist']==playlist) & (df['General_genre']=='Unknown'),
               'General_genre'] = 'African'
        
        # Hip-Hop
        if ('Hip Hop' in playlist or 'Mellow Bars' in playlist or
            'Rap UK' in playlist):
            df.loc[(df['Playlist']==playlist) & (df['General_genre']=='Unknown'),
               'General_genre'] = 'Hip hop & Rap'
        
        if ('Acid Jazz' in playlist):
            df.loc[(df['Playlist']==playlist) & (df['General_genre']!='Hip hop & Rap'),
               'General_genre'] = 'Hip hop & Rap'
        # Latin
        if ('Latino Caliente' in playlist or 'Latino Vibes' in playlist or
            'Love Reggaeton & Latino' in playlist or
            'Msica Latina' in playlist or 'Baila Reggaeton' in playlist):
            df.loc[(df['Playlist']==playlist) & (df['General_genre']!='Latin'),
               'General_genre'] = 'Latin'
      
        # Avant-garde
        if ('Avant-garde' in playlist or 'Avant-Garde' in playlist or
            'Lofi' in playlist):
            df.loc[(df['Playlist']==playlist) & (df['General_genre']=='Unknown'),
               'General_genre'] = 'Avant-garde'
        
        # Blues
        if ('Blues' in playlist or 'BLUES' in playlist or 'Nu-Blue' in playlist):
            df.loc[(df['Playlist']==playlist) & (df['General_genre']=='Unknown'),
               'General_genre'] = 'Blues'
        # Caribbean
        if ('CARIBBEAN' in playlist or 'Caribbean' in playlist):
            df.loc[(df['Playlist']==playlist) & (df['General_genre']=='Unknown'),
               'General_genre'] = 'Caribbean'
        
        # Country
        if ('Country' in playlist or 'New Boots' in playlist or
            'Texas' in playlist or 'Nashville' in playlist):
            df.loc[(df['Playlist']==playlist) & (df['General_genre']=='Unknown'),
               'General_genre'] = 'Country'
        
        # Easy listenings
        if ('Easy Listening' in playlist):
            df.loc[(df['Playlist']==playlist) & (df['General_genre']=='Unknown'),
               'General_genre'] = 'Easy listening'
        
        if ('Ambient Chill' in playlist):
            df.loc[(df['Playlist']==playlist) & (df['General_genre']!='Easy listening'),
               'General_genre'] = 'Easy listening'
        
        # ELECTRONIC
        if ('Drum & Bass' in playlist or 'mint' in playlist or 
            'Electronic' in playlist or 'Electrnica' in playlist or
            'Dance' in playlist or 'Verano Ibiza' in playlist or 
            'Main Stage' in playlist or 'Electro Sur' in playlist):
            df.loc[(df['Playlist']==playlist) & (df['General_genre']=='Unknown'),
               'General_genre'] = 'Electronic'
        # Folk    
        if ('Folk' in playlist):
            df.loc[(df['Playlist']==playlist) & (df['General_genre']=='Unknown'),
               'General_genre'] = 'Folk'
        # Rock
        if ('Funky, Heavy, Bluesy' in playlist):
            df.loc[(df['Playlist']==playlist) & (df['General_genre']!='Rock'),
               'General_genre'] = 'Rock'
            
        
        # Pop
        if ('A Cappella' in playlist):
            df.loc[(df['Playlist']==playlist) & (df['General_genre']=='Unknown'),
               'General_genre'] = 'Pop'
        if ('Club Acoustic' in playlist):
            df.loc[(df['Playlist']==playlist) & (df['General_genre']!='Pop'),
               'General_genre'] = 'Pop'
        if ('Easy Listeining' in playlist):
            df.loc[(df['Playlist']==playlist) & (df['General_genre']!='Pop'),
               'General_genre'] = 'Pop'
        # R&B
        if ('R&B' in playlist):
            df.loc[(df['Playlist']==playlist) & (df['General_genre']=='Unknown'),
               'General_genre'] = 'R&B & Soul'
        if ('contemporary r&b' in playlist):
            df.loc[(df['Playlist']==playlist) & (df['General_genre']!='R&B & Soul'),
               'General_genre'] = 'R&B & Soul'
        if ('Acid Jazz & Funk & Soulful House' in playlist):
            df.loc[(df['Playlist']==playlist) & (df['General_genre']!='R&B & Soul'),
               'General_genre'] = 'R&B & Soul'
            
        if ('Punk' in playlist):
            df.loc[(df['Playlist']==playlist) & (df['General_genre']=='Unknown'),
               'General_genre'] = 'Rock'
        
        if ('Jazz' in playlist or 'De Manhattan' in playlist or 'Butter' in playlist):
            df.loc[(df['Playlist']==playlist) & (df['General_genre']=='Unknown'),
               'General_genre'] = 'Jazz'
        
    return df

def clean_playlist_name(row):
    printable = set(string.printable)
    name_list = list(filter(lambda x: x in printable, str(row)))
    name_pl = ''.join(name_list).replace('"', '').replace(
        '\'', '').replace('|', '').replace('/', '').replace('  ', ' ')
    
    return name_pl
    
    
def main():
    print(88*'=')
    print('Genre analysis peformance')
    print(88*'=')
    filename = 'audio_features_HL.csv'
    df = read_dataframe(filename)
    df = get_genre_df(df)
    df_genre = df.copy()
    df_final = compute_final_genre(df_genre)
    
    df_final['General_genre'].replace('', 'Unknown', inplace=True)
    df_final['Genre'].replace('', 'Unknown', inplace=True)
    
    filename2 = filename.replace('.csv', '_gen.csv')
    save_dataframe(filename2, df_final)
    print(88*'#')
    print('End of the Genre Analysis peformance')
    print(88*'#')   
    
          
if __name__=='__main__':
    main()