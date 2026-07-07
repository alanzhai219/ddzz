import os
import argparse
import struct

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

class GGUFParser:
    # https://huggingface.co/docs/hub/main/en/gguf#gguf
    def __init__(self, gguf_file):
        self.m_gguf_file = gguf_file
        '''
        [  magic_number  ][    version    ][   tensor_count  ][metadata_kv_count][      metadata     ][      tensors        ]
        [     4 bytes    ][    4 bytes    ][     4 bytes     ][     4 bytes     ][   xxxxxxxxxxxxx   ][   yyyyyyyyyyyyyyy   ]
        '''
        self.m_magic_number = None
        self.m_version = None
        self.m_tensor_count = None
        self.m_metadata_kv_count = None
        self.m_metadata = None
        self.m_tensors = None

        # Read the GGUF file in binary mode
        # with open(self.m_gguf_file, 'rb') as f:
        #     self.bin_data = f.read()
        self.bin_data = open(self.m_gguf_file, 'rb')  # Reopen the binary data as a file-like object for reading

    def parse(self):
        # Implement the parsing logic here
        print(f"Parsing GGUF file: {self.m_gguf_file}")
        # Add your parsing code here
        import pdb; pdb.set_trace()  # Debugging breakpoint
        self.get_magic_number()
        self.get_version()
        self.get_tensor_count()
        self.get_metadata_kv_count()

        self.bin_data.close()
        print("FINISHED PARSING")

    def get_magic_number(self):
        # read [0:4)
        self.m_magic_number = struct.unpack('<4s', self.bin_data.read(4))[0]
    
    def get_version(self):
        # read [4:8)
        self.m_version = struct.unpack('<I', self.bin_data.read(4))[0]
    
    def get_tensor_count(self):
        # read [8:16)
        self.m_tensor_count = struct.unpack('<Q', self.bin_data.read(8))[0]

    def get_metadata_kv_count(self):
        # read [16:24)
        self.m_metadata_kv_count = struct.unpack('<Q', self.bin_data.read(8))[0]

    def get_metadata(self):

        def _read_string():
            pass
        def _read_value():
            pass

        self.m_metadata = {}
        for _ in range(self.m_metadata_kv_count):
            self.m_metadata[key] = val
        pass


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