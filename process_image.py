from datetime import datetime
from os import path, name as osname
from sys import argv, stdout
from time import sleep

import cv2
import numpy as np
from PIL import Image
from numba import njit
from pytube import YouTube

if osname == 'nt':  # enable ansi escape codes on windows cmd.exe
    from ctypes import windll
    windll.kernel32.SetConsoleMode(windll.kernel32.GetStdHandle(-11), 7)
FRAME_RATE = 20
FRAME_TIME = 1 / FRAME_RATE
GRAY_SCALE = " .:-=+*#%@"
GRAY = list(i for i in GRAY_SCALE)
DIV = 256 / len(GRAY)
# ANSI escape codes
FG_MODE = True
if FG_MODE:
    WHITE = '\033[0m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
else:
    WHITE = '\033[0m'
    RED = '\033[41m'
    GREEN = '\033[42m'
    YELLOW = '\033[43m'
    BLUE = '\033[44m'
    PURPLE = '\033[45m'
    CYAN = '\033[46m'
HIDE_CURSOR = '\033[?25l'

FAST = True if 'fast' in argv else False
DEBUG = True if 'debug' in argv else False
COLOR = True if 'color' in argv else False
COLOR_THRESHOLD = 64
MAX_X = 315
MAX_Y = 80
V_BLANK = True
IMAGE_FORMATS = ['png', 'jpg', 'bmp', 'jpeg', 'gif']
VIDEO_FORMATS = ['mp4']
AVAILABLE_COLORS = [WHITE, RED, GREEN, BLUE, YELLOW, CYAN, PURPLE]


class Timer:
    def __init__(self):
        self.start = datetime.now().timestamp()
        self.times = {}

    def end(self, r):
        self.times[r] = datetime.now().timestamp() - self.start
        self.start = datetime.now().timestamp()


def clear_screen():
    stdout.write(chr(27) + "[2J")


@njit(parallel=True, fastmath=True)
def populate_color_hsv(h, s, v):
    """Returns an approximate index value in the color list to represent the color of this pixel"""
    if 140 >= h >= 81:  # green
        if s + v >= 150:
            return 2
    elif 221 <= h <= 240:  # blue
        if s + v >= 150:
            return 3
    elif 60 >= h >= 51:  # yellow
        if s + v >= 150:
            return 4
    elif 170 <= h <= 200:  # cyan
        if s + v >= 150:
            return 5
    elif h >= 281 or h <= 320:  # purple/magenta
        if s + v >= 150:
            return 6
    elif h >= 355 or h <= 10:  # red
        if s + v >= 150:
            return 1
    return 0


@njit(parallel=True, fastmath=True)
def populate_pixel_hsv(v):
    """Returns the correct index value in the gray scale to represent the color of this value"""
    pixel_index = int(v / DIV)
    return pixel_index


def ascii_convert_fast(frame, output_x, output_y, left_blank):
    """Fast algorithm"""
    if DEBUG:
        t = Timer()
    frame = cv2.resize(frame, (output_x, output_y))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    screen = ''
    for y in range(output_y):
        line = ''
        for x in range(left_blank):
            line += GRAY[0]
        if COLOR:
            for x in range(output_x):
                h, s, v = hsv[y][x]
                p = populate_pixel_hsv(v)
                c = populate_color_hsv(h, s, v)
                line += AVAILABLE_COLORS[c] + GRAY[p]
            line += WHITE if COLOR else ''
        else:
            for x in range(output_x):
                v = hsv[y][x][2]
                line += GRAY[populate_pixel_hsv(v)]
        screen += line + '\n'
    if DEBUG:
        t.end('ascii_convert_fast')
        for data in t.times:
            screen += data + ': ' + "{0:.3f}s\t".format(t.times[data])
        screen += ' potential fps: ' + str(1 / t.times['ascii_convert_fast']) + ' output_x: ' + str(
            output_x) + ' output_y: ' + str(output_y) + ' COLOR: ' + str(COLOR)
    screen = screen.rstrip()
    if V_BLANK or output_y < MAX_Y:
        clear_screen()
    return screen


def ascii_convert(im, rgb=True, transparency_color=None):
    """Slow algorithm"""
    if DEBUG:
        t = Timer()
    im = im.resize((im.size[0] * 2, im.size[1]))
    if im.size[0] > MAX_X or im.size[1] > MAX_Y:
        if im.size[0] > MAX_X:
            percent = (MAX_X / float(im.size[0]))
            h_size = int((float(im.size[1]) * float(percent)))
            w_size = MAX_X
        if im.size[1] > MAX_Y:
            percent = (MAX_Y / float(im.size[1]))
            h_size = MAX_Y
            w_size = int((float(im.size[0]) * float(percent)))
        im = im.resize((w_size, h_size))
    pixel_access = im.load()
    left_blank = int((MAX_X - im.size[0]) / 2)

    screen = ''
    if rgb:
        hsv = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2HSV)
        for y in range(im.size[1]):
            line = ''
            for x in range(left_blank):
                line += GRAY[0]
            if COLOR:
                for x in range(im.size[0]):
                    h, s, v = hsv[y][x]
                    p = populate_pixel_hsv(v)
                    c = populate_color_hsv(h, s, v)
                    line += AVAILABLE_COLORS[c] + GRAY[p]
            else:
                for x in range(im.size[0]):
                    h, s, v = hsv[y][x]
                    line += GRAY[populate_pixel_hsv(v)]
            screen += line + '\n'
    else:  # image is indexed color
        for y in range(im.size[1]):
            line = ''
            for x in range(left_blank):
                line += GRAY[0]
            for x in range(im.size[0]):
                color_index = pixel_access[x, y]
                p = populate_pixel_hsv(color_index)
                line += GRAY[p] if color_index != transparency_color else GRAY[0]
            screen += line + '\n'

    if DEBUG:
        t.end('ascii_convert')
        for data in t.times:
            screen += data + ': ' + "{0:.3f}s\t".format(t.times[data])
        screen += ' potential fps: ' + str(1 / t.times['ascii_convert'])
    screen = screen.rstrip()
    if im.size[1] > MAX_Y:
        for line in range(MAX_Y - im.size[1]):
            screen += '\n'
    return screen


def process_video(file):
    """Used for all video input"""
    cap = cv2.VideoCapture(file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_s = total_frames / fps
    video_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if FAST:
        if video_x > MAX_X or video_y > MAX_Y:
            if video_x > MAX_X:
                percent = (MAX_X / float(video_x))
                h_size = int((float(video_y) * float(percent)))
                w_size = MAX_X
            if video_y > MAX_Y:
                percent = (MAX_Y / float(video_y))
                h_size = MAX_Y
                w_size = int((float(video_x) * float(percent)))
            output_x = w_size * 2  # multiply width by 2 because most fonts are taller than they are wide
            output_y = h_size
        else:
            output_x = MAX_X
            output_y = MAX_Y
        left_blank = int((MAX_X - output_x) / 2)
    current_frame = 0
    start_time = datetime.now().timestamp()
    t = Timer()  #
    while True:
        frame_time = datetime.now().timestamp()
        ret, frame = cap.read()
        current_frame += 1
        if not ret:
            break
        time_elapsed = frame_time - start_time
        should_be_at = round(time_elapsed * fps)
        skip_frame = True if current_frame < should_be_at else False
        if not skip_frame:
            if FAST:
                # Fast algorithm, use njit no python mode
                print(ascii_convert_fast(frame, output_x, output_y, left_blank))
            else:
                # Slow algorithm, use PIL
                f = Image.fromarray(frame)
                print(ascii_convert(f, rgb=True))
            if DEBUG:
                t.end('actual frame time')  #
                for data in t.times:
                    print(data + ': ' + "{0:.3f}s\t".format(t.times[data]), end='', flush=True)
                print('actual fps', "{0:.1f}".format(1 / (t.times['actual frame time'])), '\tcurrent_frame',
                      current_frame, '\tshould_be_at', should_be_at, '\tvideo progress %',
                      (time_elapsed / duration_s) * 100, end='', flush=True)
            frame_took = datetime.now().timestamp() - frame_time
            sleep_time = FRAME_TIME - frame_took
            if sleep_time > 0:
                sleep(sleep_time)
    print('')
    cap.release()
    cv2.destroyAllWindows()
    return


def process_youtube(link):
    """Used if argument is a link to youtube"""
    filename = link.split('=')[-1].rstrip('/')
    #  yt = YouTube(link)
    #  todo: show thumbnail while downloading
    #  yt.thumbnail_url
    if path.isfile(filename + '.mp4'):
        process_video(filename + '.mp4')
        return
    else:
        yt = YouTube(link)
        print('Downloading %s...' % yt.title)
        yt.streams.filter(file_extension='mp4', progressive=True).order_by('resolution').desc().first().download(
            filename=filename)
        process_video(filename + '.mp4')
        return


def process_image(file):
    """Used for image inputs"""
    im = Image.open(file)
    if im.format == 'GIF' and im.is_animated:
        n_frames = 0
        is_rgb = True
        trans = None
        if isinstance(im.load()[0, 0], int):
            is_rgb = False
            if 'transparency' in im.info:
                trans = im.info['transparency']
        if DEBUG:
            t = Timer()
        while im:
            frame_timer = datetime.now().timestamp()
            name = '%s-%s.png' % (file, str(n_frames))
            im.save(name, 'png')
            f = Image.open(name)
            out = ascii_convert(f, rgb=is_rgb, transparency_color=trans)
            if DEBUG:
                t.end('generate image')
            print(out)
            if DEBUG:
                t.end('print image')
                for data in t.times:
                    print(data + ': ' + "{0:.3f}s\t".format(t.times[data]), end='', flush=True)
            sleep_time = (1 / FRAME_RATE) - (datetime.now().timestamp() - frame_timer)
            sleep(sleep_time if sleep_time > 0 else 0)
            n_frames += 1
            try:
                im.seek(n_frames)
            except EOFError:
                n_frames = 0
    elif str(im.format).lower() in IMAGE_FORMATS:
        print(ascii_convert(im))


def main():
    #  look at file extension to figure out what to do with it
    print(HIDE_CURSOR)
    if len(argv) < 2:
        return print('No arguments given!')
    if argv[1].startswith('http') and 'youtube' in argv[1]:
        process_youtube(argv[1])
        return
    ext = argv[1].split('.')[-1].lower()
    if ext in IMAGE_FORMATS:
        process_image(argv[1])
    elif ext in VIDEO_FORMATS:
        process_video(argv[1])
    else:
        print('Not sure what to do with %s files' % ext)


main() if __name__ == '__main__' else exit()
