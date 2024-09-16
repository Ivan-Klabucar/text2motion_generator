import os
import torch
import numpy as np
import matplotlib
from pathlib import Path
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches

matplotlib.use('Agg') # Disable the GUI aspect of matplot lib

AMASS_joint_paths = [
    [0,2,5,8,11],
    [0,1,4,7,10],
    [0,3,6,9,12,15],
    [9,13,16,18,20,22],
    [9,14,17,19,21,23]
]

VIEW_ANGLE = [0, 8, 90]


def visualize_sequence_3d(seq, out_path, joint_paths=None):
    elev, azim, roll = VIEW_ANGLE

    def update(frame):
        ax.clear()

        ax.set_xlim3d([min_x, max_x])
        ax.set_ylim3d([min_y, max_y])
        ax.set_zlim3d([min_z, max_z])

        
        ax.view_init(elev=elev, azim=azim, roll=roll)
        ax.set_box_aspect(aspect_ratio)

        ax.set_xticklabels([])  # Remove x-axis numbers
        ax.set_yticklabels([]) 
        ax.set_zticklabels([]) 

        x = seq[frame, :, 0]
        y = seq[frame, :, 1]
        z = seq[frame, :, 2]

        if joint_paths is None: ax.scatter(x, y, z)

        if joint_paths:
            for joint_path in joint_paths:
                joint_path_coords = [seq[frame, joint, :] for joint in joint_path]
                x = [coord[0] for coord in joint_path_coords]
                y = [coord[1] for coord in joint_path_coords]
                z = [coord[2] for coord in joint_path_coords]
                ax.plot(x,y,z,color = 'g')

    min_x, min_y, min_z = np.min(seq, axis=(0, 1))
    max_x, max_y, max_z = np.max(seq, axis=(0, 1))

    x_range = max_x - min_x
    y_range = max_y - min_y
    z_range = max_z - min_z
    aspect_ratio = [x_range, y_range, z_range]

    # create the animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ani = FuncAnimation(fig, update, frames=seq.shape[0], interval=50)
    ani.save(out_path, writer='ffmpeg')




class Motion_To_Mp4_Converter:
    def __init__(self, joint_paths=AMASS_joint_paths) -> None:
        self.vid_out_dir = Path(os.getenv('VIDEO_OUPUT_DIR'))
        self.joint_paths = joint_paths

    def create_visualization(self, seq, name):
        output_path = self.vid_out_dir / name
        visualize_sequence_3d(seq, output_path, joint_paths=self.joint_paths)

amass_motion2mp4_service = Motion_To_Mp4_Converter(joint_paths=AMASS_joint_paths)