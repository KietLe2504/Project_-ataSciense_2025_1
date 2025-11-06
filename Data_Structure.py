import numpy as np
import pandas as pd
from collections import defaultdict
data = np.load("Project_-ataSciense_2025_1\youtube_comments (3000 videos).npy", allow_pickle=True)

#This data have Channel Name, Channel ID, Video Name, Video ID, Commenter, Comment
#Now take only Name_Channel
def create_channel_feature(data):
    videos = defaultdict(set)
    num_video = defaultdict(int)
    commentors = defaultdict(set)
    num_commentors = defaultdict(int)
    Channel = set()
    for info in data:
        Channel_Name, _ , Video_Name, _ ,Commentor, _ = info
        videos[Channel_Name].add(Video_Name)
        commentors[Channel_Name].add(Commentor)
        Channel.add(Channel_Name)
    
    for channel in Channel:
        num_commentors[channel] = len(commentors[channel])
        num_video[channel] = len(videos[channel])

    df = pd.DataFrame({"Number_Video" : num_video, "Number_Commentors" : num_commentors})
    df.index.name = "Channel"

    return df

def create_commentor_feature(data):
    videos = defaultdict(set)
    length_comment = defaultdict(int)
    channels = defaultdict(set)
    num_videos = defaultdict(int)
    num_channels = defaultdict(int)
    comment = set()
    for info in data:
        _, Channel_Id , _, Video_ID ,Commentor, Comment = info
        videos[Commentor].add(Video_ID)
        length_comment[Commentor]+=len(Comment)
        comment.add(Commentor)
        channels[Commentor].add(Channel_Id)
    
    for com in comment:
        num_videos[com] = len(videos[com])
        num_channels[com] = len(channels[com])

    df = pd.DataFrame({"Number_Videos" : num_videos, "Number_Channels": num_channels, "Comment_Lenght": length_comment})
    df.index.name = "Commentor"
    return df

print(create_channel_feature(data))

    







