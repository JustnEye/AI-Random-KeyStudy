def left_shift_128bit(bits, shift):
    return bits[shift:] + bits[:shift]

def generate_idea_key_schedule(user_key_bits):
    subkeys = []
    current_bits = user_key_bits.copy()
    while len(subkeys) < 52:
        for i in range(0, 128, 16):
            if len(subkeys) >= 52:
                break
            subkeys.append(current_bits[i:i+16])
        current_bits = left_shift_128bit(current_bits, 25)
    return subkeys

def to_binary_string(subkeys):
    return ''.join([''.join(bits) for bits in subkeys])

def write_nist_format(bitstring, filename, stream_length=100000, num_streams=10):
    with open(filename, "w") as f:
        for _ in range(num_streams):
            repeat_times = (stream_length + len(bitstring) - 1) // len(bitstring)
            stream = (bitstring * repeat_times)[:stream_length]
            f.write(stream + "\n")
    print(f"✅ Saved {num_streams} sequences to {filename}")

# -- Define your 128-bit user key here --
user_key_subblocks = [1, 2, 3, 4, 5, 6, 7, 8]
user_key_bits = ''.join(f'{block:016b}' for block in user_key_subblocks)
user_key_bits = list(user_key_bits)

subkeys = generate_idea_key_schedule(user_key_bits)
bitstring = to_binary_string(subkeys)
write_nist_format(bitstring, "traditional_key.txt")

