import array
import io
import json
import torch
import logging

logger = logging.getLogger(__name__)

def align_next(offset, align):
    m = offset % align
    if m == 0: return offset
    return offset + align - m

class Request:
    def __init__(self):
        pass

    def decode(self, b: bytes):
        reader = io.BytesIO(b)
        byte_size = int.from_bytes(reader.read(4), "little")
        assert int.from_bytes(reader.read(4), "little") == 0x69babe69
        block_cnt = int.from_bytes(reader.read(4), "little")
        json_size = int.from_bytes(reader.read(4), "little")

        json_str = reader.read(json_size).decode(encoding="utf-8")
        json_obj = json.loads(json_str)

        offset = reader.tell()
        padding = align_next(offset, 4) - offset
        _ = reader.read(padding) 
        assert reader.tell() % 4 == 0

        logger.info("decode message: size=%d, json_size=%d, padding=%d, block_cnt=%d", byte_size, json_size, padding, block_cnt)
        logger.info("json: %s", json_str)

        tensors = []
        for i in range(0, block_cnt):
            start = reader.tell()

            block_size = int.from_bytes(reader.read(4), "little")
            dim_cnt = int.from_bytes(reader.read(4), "little")

            dims = array.array("I")
            dims.frombytes(reader.read(4 * dim_cnt))

            elem_cnt = 1
            for x in dims: elem_cnt *= x

            data = array.array("f")
            data.frombytes(reader.read(4 * elem_cnt))
            logger.info("tensor %d: size=%d, dim_cnt=%d dims=%s", i, block_size, dim_cnt, f"{dims.tolist()}")

            assert start + block_size == reader.tell()
            t = torch.tensor(data).reshape(dims.tolist())
            tensors.append(t)



