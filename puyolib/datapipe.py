import multiprocessing as mp
import subprocess
import os
import cv2
import time
from puyolib.vision import processFrame

import timeit

VIDEO_FILE = "./dev/momoken_vs_tom2.mp4"
VIDEO_CHUNK_DEST = ".tmp/"
TEMP_FILE_PREFIX = "tmp"


def processVideo(src_filepath, start_time, end_time):
    """Return... something?"""

    # For the designated amount of time, chunk the video for each spare processor.
    no_cpu = os.cpu_count()
    _, f_ext = os.path.splitext(VIDEO_FILE)
    video_chunk_filepaths = []
    for idx in range(0, no_cpu - 1):
        tmp_file = os.path.join(VIDEO_CHUNK_DEST, TEMP_FILE_PREFIX + str(idx) + f_ext)
        start = round(idx * end_time / (no_cpu - 1))
        end = round(start + (end_time / (no_cpu - 1)))
        chunkVideo(src_filepath, tmp_file, start, end)
        video_chunk_filepaths.append(tmp_file)

    # Spawn the parallel processors and output queue.
    frame_processors = []
    data_queue = mp.Queue(maxsize=256)
    for path in video_chunk_filepaths:
        args = (path, data_queue)
        frame_processors.append(
            mp.Process(target=parallelProcessVideo, args=args, daemon=True)
        )
    for proc in frame_processors:
        proc.start()

    # Loop until processing is complete.
    data_compiled = []
    queue_complete = False
    sentinel_count = 0
    while not queue_complete:
        if data_queue.empty():
            continue
        result = data_queue.get()
        if result is None:
            sentinel_count += 1
            if sentinel_count == len(frame_processors):
                queue_complete = True
                data_queue.close()
        else:
            data_compiled.append(result)

    for proc in frame_processors:
        proc.join()
        proc.close()

    return None  # ?


def parallelProcessVideo(src_filepath, data_queue):
    """Place into the data queue the processed results of each frame."""

    cap = cv2.VideoCapture(src_filepath)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = processFrame(frame)
        data_queue.put(result)
    # Release and return None as a sentinel.
    cap.release()
    data_queue.put(None)


def chunkVideo(src_filepath, dest_filepath, start_time, end_time):
    """ Chunk the input video into a copy, units seconds."""

    cmd = (
        "ffmpeg -y -i "
        + src_filepath
        + " -ss "
        + str(start_time)
        + " -to "
        + str(end_time)
        + " "
        + dest_filepath
    )
    subp = subprocess.run(
        cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return subp.returncode


def main():
    t = timeit.Timer(lambda: processVideo(VIDEO_FILE, 0, 60))
    print(t.timeit(number=1))


if __name__ == "__main__":
    main()
