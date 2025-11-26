import array
import io

class Message:
    def __init__(self):
        # self.tensors: Map<string, (dims, data)>
        # dims: array.array("I")
        # data: array.array("f")
        self.tensors = {}

    def clear(self):
        self.tensors = {}

    def get(self, channel: str):
        return self.tensors[channel]

    def set(self, channel, dims: array.array, data: array.array):
        self.tensors[channel] = (dims, data)
    
    def encode(self):
        network = "big"
        writer = io.BytesIO()
        writer.write(int.to_bytes(len(self.tensors), 4, network))

        for channel in self.tensors.keys():
            (dims, data) = self.tensors[channel]
            enc = channel.encode(encoding="utf8")
            writer.write(int.to_bytes(len(enc), 4, network))
            writer.write(enc)

            padding = len(enc) % 4
            if padding != 0: padding = 4 - padding
            writer.write(bytes(padding))

            writer.write(int.to_bytes(len(dims), 4, network))
            writer.write(dims.tobytes())
            writer.write(data.tobytes())

        return writer.getbuffer()

    def decode(self, b: bytes):
        network = "big"
        self.tensors = {}

        reader = io.BytesIO(b)
        num_packets = int.from_bytes(reader.read(4), byteorder=network, signed=False)

        for i in range(0, num_packets):
            channel_len = int.from_bytes(reader.read(4), byteorder=network, signed=False)
            channel = reader.read(channel_len).decode(encoding="utf8")
            
            padding = channel_len % 4
            if padding != 0: padding = 4 - padding
            reader.read(padding)

            n_dim = int.from_bytes(reader.read(4), byteorder=network, signed=False)
            dims = array.array('I')
            dims.frombytes(reader.read(4 * n_dim))

            elem_cnt = 1
            for d in dims: elem_cnt *= d

            data = array.array('f')
            data.frombytes(reader.read(4 * elem_cnt))

            self.tensors[channel] = (dims, data);

