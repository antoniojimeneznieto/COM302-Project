import sys

import numpy as np
import string
import random

# Set distance between constellation points
d = 1

# Message size
message_size  = 0

def transmitter(message):
    global message_size
    message_size = len(message)
    # Add 50 X to message
    message = message + 50 * 'X'
    # Convert text message to binary representation
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    print("[TRANSMITTER] binary message:", binary_message)
    print("[TRANSMITTER] binary message size:", len(binary_message))

    # Group bits into 8-bit chunks for 256-QAM
    binary_chunks = [binary_message[i:i + 8] for i in range(0, len(binary_message), 8)]

    def generate_symbol_map(d):
        symbol_map = {}
        for i in range(16):
            for j in range(16):
                symbol_map[format(i, '04b') + format(j, '04b')] = complex((2 * i - 15) * d, (2 * j - 15) * d)
        return symbol_map

    symbol_map = generate_symbol_map(d)

    qam_signal = [symbol_map[chunk] for chunk in binary_chunks]

    # Separate the real and imaginary parts of the QAM signal
    signal_parts = [(sample.real, sample.imag) for sample in qam_signal]

    # Flatten the list of tuples into a single list
    flat_signal = [part for sample in signal_parts for part in sample]

    print("[TRANSMITTER] flat signal:", flat_signal)


    # SVD of B
    A = np.array([[11, 10], [10, 11]])
    B = np.kron(np.eye(np.size(flat_signal) // 2), A)
    U, S, V = np.linalg.svd(B, full_matrices=True)

    # Use V to transform the signal
    flat_signal_transformed = (V.T @ flat_signal)

    print("[TRANSMITTER] Vt @ flat_signal:", flat_signal_transformed)

    return flat_signal_transformed

def receiver(received_signal):
    def find_closest_point(point, symbol_map):
        # Find the constellation point closest to the received point
        distances = [abs(point - constellation_point) for constellation_point in symbol_map.keys()]
        closest_point = min(distances)
        return symbol_map[list(symbol_map.keys())[distances.index(closest_point)]]

    received_signal = np.array(received_signal)[0]
    print("[RECEIVER] received signal:", received_signal)
    print("[RECEIVER] received signal size:", len(received_signal))

    # Compute SVD of B
    A = np.array([[11, 10], [10, 11]])
    B = np.kron(np.eye(np.size(received_signal) // 2), A)

    U, S, V = np.linalg.svd(B, full_matrices=True)

    # Compute pseudo-inverse of B
    B_inv = np.diag(1 / S) @ U.T

    # Apply the pseudo-inverse of B to the received signal
    equalized_signal = B_inv @ received_signal

    print("[RECEIVER] equalized signal:", equalized_signal)

    def generate_symbol_map(d):
        symbol_map = {}
        for i in range(16):
            for j in range(16):
                symbol_map[format(i, '04b') + format(j, '04b')] = complex((2 * i - 15) * d, (2 * j - 15) * d)
        return symbol_map

    def reverse_dict(dictionary):
        reversed_dict = {value: key for key, value in dictionary.items()}
        return reversed_dict

    symbol_map = generate_symbol_map(d)
    symbol_map = reverse_dict(symbol_map)


    qam_signal = [complex(equalized_signal[i], equalized_signal[i + 1]) for i in range(0, len(equalized_signal), 2)]

    print("[RECEIVER] received signal QAM after equalizing:", qam_signal)


    demodulated_signal = [find_closest_point(point, symbol_map) for point in qam_signal]

    # Convert binary message back to text
    binary_message = ''.join(demodulated_signal)
    print("[RECEIVER] binary received signal demodulated:", binary_message)
    text_message = ''.join(chr(int(binary_message[i:i + 8], 2)) for i in range(0, len(binary_message), 8))

    # Return the first message_size characters of the text message (we don't want the added Xs)
    return text_message[:message_size]


def channel(sent_signal):
    sent_signal = np.array(sent_signal)
    assert np.size(sent_signal) <= 200, "n must be <= 100"
    n = np.size(sent_signal) // 2
    x = sent_signal[0:2 * n]
    s = sum(x ** 2) / np.size(x)
    print("S: ", s)
    sigma = 1
    if s > 1:
        sigma = np.sqrt(s)
    print("sigma: ", sigma)
    Z = np.random.normal(0, sigma, size=(2 * n, 1))
    A = np.array([[11, 10], [10, 11]])
    B = np.kron(np.eye(n), A)
    print("B:", B)
    print("B dot x: ", B.dot(x))
    print("Z:", Z)
    Y = B.dot(x) + Z.T
    print("Y:", Y)
    return Y

def generate_random_string(length):
    # All ASCII characters
    ascii_characters = string.ascii_letters + string.digits + string.punctuation
    # Generate the random string
    random_string = ''.join(random.choice(ascii_characters) for _ in range(length))
    return random_string


def example_usage():
    # Example usage:
    message = generate_random_string(50)

    X = transmitter(message)  # Encode our message
    Y = channel(X)  # Simulate the treatment done by the channel
    reconstructed_message = receiver(Y)  # Decode the message received by the channel

    print("Original message:", message)
    print("Reconstructed message:", reconstructed_message)

    return message != reconstructed_message


def save_transmitted_signal_to_file(message, filename):
    # Transmit the message
    qam_signal = transmitter(message)

    # Save the signal to the file
    with open(filename, 'w') as f:
        for sample in qam_signal:
            f.write(str(sample) + '\n')


def receive_message_from_file(filename):
    # Read the file
    with open(filename, 'r') as f:
        signal_parts = [float(line.strip()) for line in f]

    # Pair up the real and imaginary parts to recreate the complex symbols
    qam_signal = [signal_parts for i in range(0, len(signal_parts), 2)]
    print("[READER] Signal:", qam_signal)
    # Pass the symbols to the receiver function
    message = receiver(qam_signal)

    return message


if __name__ == '__main__':
    message = generate_random_string(50)
    message = "You can write here :)"

    save_transmitted_signal_to_file(message, "input.txt")
    reconstructed_signal = receive_message_from_file("output.txt")

    print("length of message:",len(message))
    print("Original:", message)
    print("Reconstructed:", reconstructed_signal)
    print(message == reconstructed_signal)


    total_errors = 0
    for x in range(0):
        total_errors += example_usage()
        print("")
    print("Total messages error:", total_errors)