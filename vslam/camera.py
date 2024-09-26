import pyrealsense2 as rs
import numpy as np
import cv2


class Camera:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # Start streaming
        self.pipeline.start(self.config)

    def __call__(self, GRAY=False, DEPTH=False):
        frames = self.pipeline.wait_for_frames()
        depth_frame = None
        color_frame = frames.get_color_frame()

        if DEPTH:
            depth_frame = frames.get_depth_frame()

        if GRAY:
            color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        return color_frame, depth_frame
