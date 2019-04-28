from PIL import Image
from datetime import datetime
from time import sleep
from sys import argv, stdout
from pytube import YouTube
from os import path
from numba import njit
import cv2

FRAMERATE = 20
FRAMETIME = 1 / FRAMERATE
FRAMERATE_NTSC = 30
FRAMETIME_NTSC = 1 / FRAMERATE_NTSC
GRAYSCALE = " .:-=+*#%@"
GRAY = list(i for i in GRAYSCALE)
DIV = 256 / len(GRAY)
WHITE = '\x1b[0m'
RED = '\033[31m'
GREEN = '\033[32m'
BLUE = '\033[34m'
PURPLE = '\033[35m'
YELLOW = '\033[93m'
CYAN = '\033[36m'

COLOR_THRESHOLD = 64
MAXX = 315
MAXY = 80
COLOR = True if len(argv) > 2 and argv[2] == 'color' else False
VBLANK = True
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


def process_image(file, debug=False):
    """Used for image inputs"""
    im = Image.open(file)
    if im.format == 'GIF' and im.is_animated:
        nframes = 0
        if debug:
            t = Timer()
        while im:
            frame_timer = datetime.now().timestamp()
            if debug:
                t.end('sleep')
            name = '%s-%s.png' % (file, str(nframes))
            im.save(name, 'png')
            if debug:
                t.end('save individual frame')
            f = Image.open(name)
            if debug:
                t.end('open image')
            # clear_screen()
            if debug:
                t.end('clear screen')
            out = ascii_convert(f)
            if debug:
                t.end('generate image')
            print(out)
            if debug:
                t.end('print image')
                for data in t.times:
                    print(data + ': ' + "{0:.3f}s\t".format(t.times[data]), end='', flush=True)
                t.end('print debug')
            sleep_time = (1 / FRAMERATE) - (datetime.now().timestamp() - frame_timer)
            sleep(sleep_time if sleep_time > 0 else 0)
            nframes += 1
            try:
                im.seek(nframes)
            except EOFError:
                nframes = 0
    elif str(im.format).lower() in IMAGE_FORMATS:
        while im:
            frame_timer = datetime.now().timestamp()
            print(ascii_convert(im))
            sleep_time = (1 / FRAMERATE) - (datetime.now().timestamp() - frame_timer)
            sleep(sleep_time if sleep_time > 0 else 0)


@njit(parallel=True, fastmath=True)
def populate_pixel_rgb(y, x, frame):
    """Used by fast algorithm"""
    rgb = frame[y][x]
    pixel_index = int((rgb[0] + rgb[1] + rgb[2]) / 3 / DIV)
    return pixel_index


@njit(parallel=True, fastmath=True)
def populate_color_hsv(y, x, frame):
    """Used by fast algorithm"""
    h, s, v = frame[y][x]
    while True:
        if h >= 355 or h <= 10:  # red
            if s >= 50:
                color = 1
                break
        elif 140 >= h >= 81:  # green
            if s >= 50:
                color = 2
                break
        elif 221 <= h <= 240:  # blue
            if s >= 50:
                color = 3
                break
        elif 60 >= h >= 51:  # yellow
            if s >= 50:
                color = 4
                break
        elif 170 <= h <= 200:  # cyan
            if s >= 50:
                color = 5
                break
        elif h >= 281 or h <= 320:  # purple/magenta
            if s >= 50:
                color = 6
                break
        else:
            color = 0
            break
    return color


@njit(parallel=True, fastmath=True)
def populate_pixel_hsv(y, x, frame):
    """Used by fast algorithm"""
    h, s, v = frame[y][x]
    pixel_index = int(v / DIV)
    return pixel_index


def ascii_convert_fast(frame, output_x, output_y, left_blank, debug=False):
    """Fast algorithm"""
    if debug:
        t = Timer()
    frame = cv2.resize(frame, (output_x, output_y))
    if COLOR:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    screen = ''
    for y in range(output_y):
        line = ''
        for x in range(left_blank):
            line += GRAY[0]
        if COLOR:
            for x in range(output_x):
                p = populate_pixel_hsv(y, x, hsv)
                c = populate_color_hsv(y, x, hsv)
                line += AVAILABLE_COLORS[c] + GRAY[p]
            line += WHITE if COLOR else ''
        else:
            for x in range(output_x):
                line += GRAY[populate_pixel_rgb(y, x, frame)]
        screen += line + '\n'

    if debug:
        t.end('ascii_convert_fast')
        for data in t.times:
            screen += data + ': ' + "{0:.3f}s\t".format(t.times[data])
        screen += ' potential fps: ' + str(1 / t.times['ascii_convert_fast']) + ' output_x: ' + str(
            output_x) + ' output_y: ' + str(output_y) + ' COLOR: ' + str(COLOR)
    screen = screen.rstrip()
    if VBLANK or output_y < MAXY:
        clear_screen()
    return screen


def process_video(file, fast_algo=False, debug=False):
    """Used for all video input"""
    cap = cv2.VideoCapture(file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_s = total_frames / fps
    video_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if fast_algo:
        if video_x > MAXX or video_y > MAXY:
            if video_x > MAXX:
                percent = (MAXX / float(video_x))
                hsize = int((float(video_y) * float(percent)))
                wsize = MAXX
            if video_y > MAXY:
                percent = (MAXY / float(video_y))
                hsize = MAXY
                wsize = int((float(video_x) * float(percent)))
            output_x = wsize * 2
            output_y = hsize
        else:
            output_x = MAXX
            output_y = MAXY
        left_blank = int((MAXX - output_x) / 2)

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
            if fast_algo:
                # Fast algo, use njit no python mode
                print(ascii_convert_fast(frame, output_x, output_y, left_blank, debug))
            else:
                # Slow algo, use PIL
                f = Image.fromarray(frame)
                print(ascii_convert(f, debug))
            if debug:
                t.end('actual frametime')  #
                for data in t.times:
                    print(data + ': ' + "{0:.3f}s\t".format(t.times[data]), end='', flush=True)
                print('actual fps', "{0:.1f}".format(1 / (t.times['actual frametime'])), '\tcurrent_frame',
                      current_frame, '\tshould_be_at', should_be_at, '\tvideo progress %',
                      (time_elapsed / duration_s) * 100, end='', flush=True)
            frame_took = datetime.now().timestamp() - frame_time
            sleep_time = FRAMETIME - frame_took
            if sleep_time > 0:
                sleep(sleep_time)

    print('')
    cap.release()
    cv2.destroyAllWindows()
    return


def decide_color(r, g, b):
    """Used by slow algo"""
    if r > g and r > b:
        others = (g + b) / 2
        if r - others > COLOR_THRESHOLD:
            return RED
        else:
            return WHITE
    elif g > r and g > b:
        others = (r + b) / 2
        if g - others > COLOR_THRESHOLD:
            return GREEN
        else:
            return WHITE
    elif b > r and b > g:
        others = (r + g) / 2
        if b - others > COLOR_THRESHOLD:
            return BLUE
        else:
            return WHITE
    else:
        return WHITE


def decide_pixel(r, g, b):
    """Used by slow algo"""
    pixel = GRAY[int((r + g + b) / 3 / DIV)]
    return pixel


def populate_line(img_x, y, pixel_access, left_blank):
    """Used by slow algo"""
    line = ''
    for x in range(left_blank):
        line += GRAY[0]
    for x in range(img_x):
        r, g, b = pixel_access[x, y]
        line += decide_color(r, g, b) + decide_pixel(r, g, b) if COLOR else decide_pixel(r, g, b)
    line += WHITE if COLOR else ''
    return line + '\n'


def populate_line_int(img_x, y, pixel_access, transparency_color, left_blank):
    """Used by slow algo"""
    line = ''
    for x in range(left_blank):
        line += GRAY[0]
    for x in range(img_x):
        rgb = pixel_access[x, y]
        pixel = GRAY[int(rgb / DIV)] if rgb != transparency_color else GRAY[0]
        line += pixel
    return line + '\n'


def ascii_convert(im, debug=False):
    """Slow algo"""
    if debug:
        t = Timer()
    im = im.resize((im.size[0] * 2, im.size[1]))
    if im.size[0] > MAXX or im.size[1] > MAXY:
        if im.size[0] > MAXX:
            percent = (MAXX / float(im.size[0]))
            hsize = int((float(im.size[1]) * float(percent)))
            wsize = MAXX
        if im.size[1] > MAXY:
            percent = (MAXY / float(im.size[1]))
            hsize = MAXY
            wsize = int((float(im.size[0]) * float(percent)))
        im = im.resize((wsize, hsize))

    pixel_access = im.load()
    left_blank = int((MAXX - im.size[0]) / 2)

    is_rgb = True
    try:
        int(pixel_access[0, 0])
        is_rgb = False
        try:
            transp = im.info['transparency']
        except:
            transp = None
    except:
        pass

    screen = ''
    if is_rgb:
        for y in range(im.size[1]):
            screen += populate_line(im.size[0], y, pixel_access, left_blank)
    else:
        for y in range(im.size[1]):
            screen += populate_line_int(im.size[0], y, pixel_access, transp, left_blank)
    if debug:
        t.end('ascii_convert')
        for data in t.times:
            screen += data + ': ' + "{0:.3f}s\t".format(t.times[data])
        screen += ' potential fps: ' + str(1 / t.times['ascii_convert'])
    screen = screen.rstrip()
    if im.size[1] > MAXY:
        for line in range(MAXY - im.size[1]):
            screen += '\n'
    return screen


def process_youtube(link, fast_algo=False, debug=False):
    """Used if argument is a link to youtube"""
    filename = link.split('=')[-1].rstrip('/')
    #  yt = YouTube(link)
    #  todo: show thumbnail while downloading
    #  yt.thumbnail_url
    if path.isfile(filename + '.mp4'):
        process_video(filename + '.mp4', fast_algo, debug)
        return
    else:
        yt = YouTube(link)
        print('Downloading %s...' % yt.title)
        yt.streams.filter(file_extension='mp4', progressive=True).order_by('resolution').desc().first().download(
            filename=filename)
        process_video(filename + '.mp4', fast_algo, debug)
        return


def main():
    #  look at file extension to figure out what to do with it
    if len(argv) == 3:
        fast_algo = True
        debug = False
    elif len(argv) > 3:
        fast_algo = True
        debug = True
    else:
        fast_algo, debug = False, False
    if len(argv) < 2:
        return print('No arguments given!')
    if argv[1].startswith('http') and 'youtube' in argv[1]:
        process_youtube(argv[1], fast_algo, debug)
        return

    ext = argv[1].split('.')[-1].lower()
    if ext in IMAGE_FORMATS:
        process_image(argv[1], debug)
    elif ext in VIDEO_FORMATS:
        process_video(argv[1], fast_algo, debug)
    else:
        print('Not sure what to do with %s files' % ext)


main() if __name__ == '__main__' else exit()

