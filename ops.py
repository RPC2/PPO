import numpy as np
import imageio
from skimage.transform import resize


def set_init_state(screen):
    state = np.stack((screen, screen, screen, screen), axis=0)
    return state


def generate_gif(frame_number, frames_for_gif, reward, path, episode):
    for idx, frame_idx in enumerate(frames_for_gif):
        frames_for_gif[idx] = resize(frame_idx, (420, 320, 3),
                                     preserve_range=True, order=0).astype(np.uint8)

    imageio.mimsave(f'{path}{"ATARI_episode_{0}_frame_{1}_reward_{2}.gif".format(episode, frame_number, reward)}',
                    frames_for_gif, duration=1 / 30)
    print("GIF for episode %d saved." % episode)
