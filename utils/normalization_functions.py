import pandas as pd
import re
import numpy as np

from cleanco import cleanco
from string_grouper import group_similar_strings
from transformers import pipeline


def get_category(list):
    """ Accepts user's input strings as a list and returns dataframe
    with an entity type categorization for each string.

    Parameters
    ----------
    list : list of user's input strings

    Returns
    -------
    dataframe : dataframe with columns ['entity', 'entity_type'] and each entity
                categorized into ['serial number', 'street address', 'city or country', 'physical goods', 'company']

    """
    # Hugging Face zero shot classification pipeline
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    candidate_labels = ['serial number', 'street address', 'city or country', 'physical goods', 'company']
    entity_type = [classifier(i, candidate_labels)['labels'][0] for i in list]
    dataframe = pd.DataFrame(np.column_stack([list, entity_type]), columns=['entity', 'entity_type'])
    return dataframe


def preprocess(dataframe):
    """ Accepts dataframe of categorized entities and returns dataframe
    with an extra column for the cleaned entity.

    Parameters
    ----------
    dataframe : dataframe with columns ['entity', 'entity_type']

    Returns
    -------
    dataframe : dataframe with columns ['entity', 'entity_type', 'clean_entity']

    """
    # General preprocessing steps
    remove_special_characters = lambda x : re.sub(r'[^a-zA-z0-9\s]', '', x)
    lower_strip = lambda x : x.lower().strip().replace('  ', ' ')

    # Preprocessing steps for serial nums and company names
    remove_whitespace_serial_nums_only = lambda x : x.replace(' ', '')
    remove_legal_terms_companies_only = lambda x : cleanco(x).clean_name()

    # Map preprocessing steps to dataframe
    dataframe['clean_entity'] = dataframe['entity'].map(remove_special_characters).map(lower_strip)
    dataframe.loc[dataframe['entity_type'] == 'serial number', 'clean_entity'] = dataframe['clean_entity'].map(remove_whitespace_serial_nums_only)
    dataframe.loc[dataframe['entity_type'] == 'company', 'clean_entity'] = dataframe['clean_entity'].map(remove_legal_terms_companies_only)
    return dataframe


def entity_matcher(dataframe, min_similarity):
    """ Accepts dataframe of categorized, preprocessed entities and returns dataframe
    with entities grouped by similarity using the string_grouper library
    which uses tf-idf to calculate cosine similarities within a list.

    Parameters
    ----------
    dataframe : dataframe with columns ['entity', 'entity_type', 'clean_entity']
    min_similarity : float between 0 and 1
                     defines minimum threshold for cosine similarity to group entities

    Returns
    -------
    dataframe : dataframe with columns ['unique_entity_ID', 'unique_entity', 'entity', 'entity_type', 'clean_entity']

    """
    dataframe[['unique_entity_ID', 'unique_entity']] = group_similar_strings(dataframe['clean_entity'], min_similarity = min_similarity)
    dataframe = dataframe.groupby(['unique_entity_ID', 'unique_entity'])['entity'].apply('; '.join).reset_index()
    return dataframe


def clustering_func(dataframe):
    """ Accepts dataframe of categorized, preprocessed entities and returns dataframe
    with entities grouped by similarity using the entity_matcher function with
    a minimum similarity threshold dependant on entity type

    Parameters
    ----------
    dataframe : dataframe with columns ['entity', 'entity_type', 'clean_entity']

    Returns
    -------
    dataframe : dataframe with columns ['unique_entity_ID', 'unique_entity', 'entity']

    """
    # Define entity types with lower and higher thresholds
    low_threshold_entity_types = ['city or country', 'physical goods', 'company']
    high_threshold_entity_types = ['serial number', 'street address']

    # Subset dataframe based on threshold
    low_threshold_subset = dataframe[dataframe.entity_type.isin(low_threshold_entity_types)].copy()
    high_threshold_subset = dataframe[dataframe.entity_type.isin(high_threshold_entity_types)].copy()

    # If they are any, group entities with lower threshold of 20%
    if low_threshold_subset.empty:
        dataframe_low_threshold = low_threshold_subset
    else:
        dataframe_low_threshold = entity_matcher(low_threshold_subset, min_similarity = 0.2)

    # If they are any, group entities with higher threshold of 99%
    if high_threshold_subset.empty:
        dataframe_high_threshold = high_threshold_subset
    else:
        dataframe_high_threshold = entity_matcher(high_threshold_subset, min_similarity = 0.99)

    # Return dataframe of grouped entities
    dataframe = pd.concat([dataframe_low_threshold, dataframe_high_threshold], ignore_index = True)
    return dataframe
