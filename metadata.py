"""
This code is used to retrieve video metadata, for instance, it is used for HiEve video data.
"""

import os
from pymediainfo import MediaInfo


class Metadata:
    def __init__(self, video_path):
        media_info = MediaInfo.parse(video_path)
        start = video_path.rfind('/')
        end = video_path.find('.')

        self.video_name = video_path[start+1:end]

        self.video_data = [ track for track in media_info.tracks if track.track_type == 'Video' ]
        if len(self.video_data) > 0:
            data = self.video_data[0]
            self.fps = data.frame_rate
            if self.fps is not None:
                self.fps = int(round(float(data.frame_rate)))
            self.rotation = data.rotation
            if self.rotation is not None:
                self.rotation = int(float(data.rotation))
            self.width = int(data.width)
            self.height = int(data.height)
            self.frame_size = (self.width, self.height)
            self.frame_count = int(data.frame_count)
        else:
            print("Error loading video metadata")
            self.frame_rate = None
            self.rotation = None
            self.width = None
            self.height = None
                

if __name__ == "__main__":
    test_video_path = './data/HiEve/HIE20test/test/20.mp4'
    metadata = Metadata(test_video_path)

    print('Frame count:', metadata.frame_count)
    print('fps: ', metadata.fps)
    print('width: ', metadata.width)
    print('height: ', metadata.height)


