import multiprocessing as mp
import subprocess
import cv2
import timeit
import numpy as np
import os


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

    # Spawn the parallel processors and output queue to analyze each video chunk.
    frame_processors = []
    data_queues = []
    for path in video_chunk_filepaths:
        data_queue = mp.Queue(maxsize=10000)
        args = (path, data_queue)
        frame_processors.append(
            mp.Process(target=parallelProcessVideo, args=args, daemon=True)
        )
        data_queues.append(data_queue)
    for proc in frame_processors:
        proc.start()

    # Loop until complete processing is complete.
    incomplete = True
    while incomplete:
        incomplete = any([proc.is_alive() for proc in frame_processors])

    # Loop through queues to collect data.
    data_compiled = []
    incomplete = True
    while incomplete:
        for dq in data_queues:
            data_compiled.append(dq.get_nowait())
        incomplete = any([not dq.empty() for dq in data_queues])

    # Tear down.
    for proc in frame_processors:
        proc.close()
    for dq in data_queues:
        dq.close()

    return None  # ?


def parallelProcessVideo(src_filepath, data_queue):
    """Place into the data queue the processed results of each frame."""

    cap = cv2.VideoCapture(src_filepath)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Process the frame.
        for _ in range(0, 10):
            frame = cv2.blur(frame, (5, 5))
        result = np.amax(frame)
        data_queue.put(result)
    # Release and return.
    cap.release()


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
