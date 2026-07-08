import os
import argparse
import struct

# spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
# blog: https://tlbflush.org/post/2025_02_17_gguf_weekend/

VALUE_FORMATS = {
    0: "B",  # UINT8
    1: "b",  # INT8
    2: "H",  # UINT16
    3: "h",  # INT16
    4: "I",  # UINT32
    5: "i",  # INT32
    6: "f",  # FLOAT32
    7: "?",  # BOOL
    10: "Q",  # UINT64
    11: "q",  # INT64
    12: "d",  # FLOAT64
}

class GGUFString_t:
    def __init__(self, len, string):
        self.len = len
        self.value = string

class GGUFMetadata_value_t:
    pass

class GGUFMetadataKV_t:
    def __init__(self, key, value_type, value):
        self.key = key
        self.value_type = value_type
        self.value = value

class GGUFHeader:
    def __init__(self, magic_number, version, tensor_count, metadata_kv_count, metadata_kv):
        self.magic_number = magic_number
        self.version = version
        self.tensor_count = tensor_count
        self.metadata_kv_count = metadata_kv_count
        self.metadata_kv = metadata_kv

class GGUFParser:
    def __init__(self, gguf_file):
        self.m_gguf_file = gguf_file
        '''
        [  magic_number  ][    version    ][   tensor_count  ][metadata_kv_count][      metadata     ][      tensors        ]
        [     4 bytes    ][    4 bytes    ][     4 bytes     ][     4 bytes     ][   xxxxxxxxxxxxx   ][   yyyyyyyyyyyyyyy   ]
        '''
        self.m_gguf_header = None
        self.m_gguf_tensor_info = None
        self.m_gguf_padding = None
        self.m_gguf_tensor_data = None

        # Read the GGUF file in binary mode
        # with open(self.m_gguf_file, 'rb') as f:
        #     self.bin_data = f.read()
        self.bin_data = open(self.m_gguf_file, 'rb')  # Reopen the binary data as a file-like object for reading
    
    def read_gguf_header(self):
        # Read the GGUF header from the binary data
        # @magic_number read [0:4)
        magic_number = struct.unpack('<4s', self.bin_data.read(4))[0]
        # @version read [4:8)
        version = struct.unpack('<I', self.bin_data.read(4))[0]
        # @tensor_count read [8:16)
        tensor_count = struct.unpack('<Q', self.bin_data.read(8))[0]
        # @metadata_kv_count read [16:24)
        metadata_kv_count = struct.unpack('<Q', self.bin_data.read(8))[0]

        def _read_string(bin_data):
            len = struct.unpack("<Q", bin_data.read(8))[0]
            string = struct.unpack(f"<{len}s", bin_data.read(len))[0]
            return GGUFString_t(len, string)

        def _read_value_type(bin_data):
            value_type = struct.unpack("<I", bin_data.read(4))[0]
            return value_type

        def _read_value():
            pass

        def _read_metadata_kv(bin_data, count):
            metadata_kv_list = []
            for _ in range(count):
                key = _read_string(bin_data)
                value_type = _read_value_type(bin_data)
                value = _read_value()
                metadata_kv_list.append(GGUFMetadataKV_t(key, value_type, value))
            return metadata_kv_list

        metadata_kv = _read_metadata_kv(self.bin_data, metadata_kv_count)

        # Create a GGUFHeader object to store the parsed header information
        self.m_gguf_header = GGUFHeader(
            magic_number=magic_number,
            version=version,
            tensor_count=tensor_count,
            metadata_kv_count=metadata_kv_count,
            metadata_kv=metadata_kv
        )

    def parse(self):
        # Implement the parsing logic here
        print(f"Parsing GGUF file: {self.m_gguf_file}")
        # Add your parsing code here
        import pdb; pdb.set_trace()  # Debugging breakpoint
        self.read_gguf_header()

        self.bin_data.close()
        print("FINISHED PARSING")

    def get_metadata(self):

        def _read_string():
            pass
        def _read_value():
            pass

        self.m_metadata_info = {}
        for _ in range(self.m_metadata_kv_count):
            key = _read_string()
            val = _read_value()
            self.m_metadata_info[key] = val


    def get_tensors(self):
        pass
        # return self.m_tensors

def main():
    parser = argparse.ArgumentParser(description="GGUF Parser")
    parser.add_argument('--model', required=True, type=str, help='Path to the GGUF model file')
    args = parser.parse_args()

    gguf_file = args.model

    gguf_parser = GGUFParser(gguf_file)
    gguf_parser.parse()

if __name__ == "__main__":
    main()