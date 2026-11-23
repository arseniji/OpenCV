"""Чтение официального ZIP по HTTP Range без хранения полного архива."""

from functools import lru_cache
from io import SEEK_CUR, SEEK_END, SEEK_SET
from urllib.request import Request, urlopen


UAV123_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=0B6sQMCU1i4NbNGxWQzRVak5yLWs&export=download&confirm=t"
    "&resourcekey=0-IjwQcWEzP2x3ec8kXtLBpA"
)


class RemoteZip:
    block_size = 2 * 1024 * 1024

    def __init__(self, url):
        self.url = url
        self.position = 0
        self.size = None
        self.fetch(0, 0)
        self.block = lru_cache(maxsize=8)(self.fetch)

    def fetch(self, start, end):
        request = Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        for attempt in range(3):
            try:
                with urlopen(request, timeout=25) as response:
                    content_range = response.headers.get("Content-Range", "")
                    prefix = f"bytes {start}-{end}/"
                    if response.status != 206 or not content_range.startswith(prefix):
                        raise ValueError("Сервер не вернул запрошенный диапазон ZIP")
                    size = int(content_range[len(prefix):])
                    if self.size is not None and self.size != size:
                        raise ValueError("Размер удалённого архива изменился")
                    payload = response.read(end - start + 2)
                    if len(payload) != end - start + 1:
                        raise OSError("Неполный ответ сервера")
                    self.size = size
                    return payload
            except OSError:
                if attempt == 2:
                    raise

    def tell(self):
        return self.position

    def seekable(self):
        return True

    def seek(self, offset, whence=SEEK_SET):
        base = {SEEK_SET: 0, SEEK_CUR: self.position, SEEK_END: self.size}[whence]
        if base + offset < 0:
            raise OSError("Позиция перед началом ZIP")
        self.position = base + offset
        return self.position

    def read(self, size=-1):
        limit = self.size if size < 0 else min(self.size, self.position + size)
        chunks = []
        while self.position < limit:
            start = self.position // self.block_size * self.block_size
            end = min(self.size, start + self.block_size)
            next_position = min(end, limit)
            chunks.append(self.block(start, end - 1)[self.position - start:next_position - start])
            self.position = next_position
        return b"".join(chunks)
