import multiprocessing as mp
import threading
import subprocess
import os
import cv2
from collections import deque as Deque
from puyolib.vision import processFrame

import timeit

VIDEO_FILE = "./dev/momoken_vs_tom2.mp4"
VIDEO_CHUNK_DEST = ".tmp/"
TEMP_FILE_PREFIX = "tmp"

CPU_COUNT = 5


def processVideo(src, start, end):
    """Return... something?"""

    video_chunk_filepaths = makeVideoChunks(src, CPU_COUNT, start, end)

    # Spawn the parallel processors and output queue.
    frame_processors = []
    data_queue = mp.Queue()
    for path in video_chunk_filepaths:
        frame_processors.append(
            mp.Process(target=processChunk, args=(path, data_queue), daemon=True)
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
        if result is not None:
            data_compiled.append(result)
        else:
            sentinel_count += 1
            if sentinel_count == len(frame_processors):
                queue_complete = True
                data_queue.close()

    for proc in frame_processors:
        proc.join()
        proc.close()

    return None  # ?


def readFrames(src, deque):
    """Read video frames into a deque."""

    cap = cv2.VideoCapture(src)
    while True:
        ret, frame = cap.read()
        if ret:
            deque.appendleft(frame)
        else:
            break
    cap.release()
    deque.appendleft(None)


def processChunk(src, data_queue):
    """Return the processed results of each frame in the video chunk."""

    frame_deque = Deque()
    frame_thread = threading.Thread(
        target=readFrames, args=(src, frame_deque), daemon=True
    )
    frame_thread.start()

    while True:
        if frame_deque:
            frame = frame_deque.pop()
            if frame is not None:
                result = processFrame(frame)
                data_queue.put(result)
            else:
                break
    data_queue.put(None)
    frame_thread.join()


def chunkVideo(src, dest, start, end):
    """Chunk the input video into a copy (with FFMPEG), units seconds."""

    start, end = tuple(map(str, (start, end)))
    args = ["ffmpeg", "-y", "-i", src, "-ss", start, "-to", end, dest]
    subp = subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return subp.returncode


def makeVideoChunks(src, n, start, end):
    """Return temporary filepaths to chunked videos."""

    _, f_ext = os.path.splitext(src)
    video_chunks = []
    for idx in range(n):
        tmp_file = TEMP_FILE_PREFIX + str(idx) + f_ext
        tmp_file = os.path.join(VIDEO_CHUNK_DEST, tmp_file)
        cstart = round(idx * end / n)
        cend = round(cstart + (end / n))
        chunkVideo(src, tmp_file, cstart, cend)
        video_chunks.append(tmp_file)
    return video_chunks


def main():
    t = timeit.Timer(lambda: processVideo(VIDEO_FILE, 0, 20))
    print(t.timeit(number=1))


if __name__ == "__main__":
    main()
