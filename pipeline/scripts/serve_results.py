"""Read-only localhost result viewer with MP4 range requests for seeking."""
import argparse
import functools
import os
import re
import shutil
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class ResultHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.byte_range = None
        path = self.translate_path(self.path)
        if os.path.isdir(path) or not self.headers.get("Range"):
            return super().send_head()
        if not Path(path).resolve().is_relative_to(Path(self.directory).resolve()):
            self.send_error(403)
            return None
        try:
            stream = open(path, "rb")
        except OSError:
            self.send_error(404)
            return None
        size = os.fstat(stream.fileno()).st_size
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers["Range"].strip())
        if not match or not any(match.groups()):
            stream.close()
            self.send_error(416)
            return None
        begin, end = match.groups()
        if begin:
            start, stop = int(begin), min(int(end) if end else size - 1, size - 1)
        else:
            start, stop = max(0, size - int(end)), size - 1
        if start >= size or stop < start:
            stream.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        self.send_response(206)
        self.send_header("Content-type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{stop}/{size}")
        self.send_header("Content-Length", str(stop - start + 1))
        self.end_headers()
        stream.seek(start)
        self.byte_range = (start, stop)
        return stream

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def copyfile(self, source, outputfile):
        try:
            if self.byte_range is None:
                shutil.copyfileobj(source, outputfile)
            else:
                remaining = self.byte_range[1] - self.byte_range[0] + 1
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    outputfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    if not directory.is_dir():
        parser.error("directory must be a folder")
    handler = functools.partial(ResultHandler, directory=str(directory))
    ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()
