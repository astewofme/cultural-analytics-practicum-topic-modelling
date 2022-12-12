#prepreration

%%capture
!pip install bertopic

import re
import pandas as pd
from datetime import datetime

# Load data
wsb = pd.read_csv('')

# convert title column in the dataframe to list of strings and format created_utc to datetime
title=wsb.title.to_list()
titles=[]
for t in title:
    t=str(t)
    titles.append(t)
wsb['titles']=pd.Series(titles)
timestamps_datetime=pd.to_datetime(wsb.created_utc, unit='s')


# import topic modelling library
from bertopic import BERTopic
topic_model = BERTopic(min_topic_size=35, verbose=True)

# fit the model
topics, _ = topic_model.fit_transform(titles)

# get the topics and their frequency
freq = topic_model.get_topic_info(); freq.head(10)

# plot the topics
fig = topic_model.visualize_topics(); fig


# visualize the topics over time
topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=20)
