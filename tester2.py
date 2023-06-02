import sys

import numpy as np
import string
import random


def transmitter(message):
    # Convert text message to binary representation
    # Add 50 X to message
    message = message + 50 * 'X'
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    print("[TRANSMITTER] binary message:", binary_message)
    print("[TRANSMITTER] binary message size:", len(binary_message))

    # Group bits into 4-bit chunks for 16-QAM
    binary_chunks = [binary_message[i:i + 8] for i in range(0, len(binary_message), 8)]

    # 16-QAM modulation
    d = 1
    symbol_map = {'0000': complex(d, d),
                  '0001': complex(d, 3 * d),
                  '0010': complex(3 * d, d),
                  '0011': complex(3 * d, 3 * d),
                  '0100': complex(-d, d),
                  '0101': complex(-d, 3 * d),
                  '0110': complex(-3 * d, d),
                  '0111': complex(-3 * d, 3 * d),
                  '1000': complex(d, -d),
                  '1001': complex(d, -3 * d),
                  '1010': complex(3 * d, -d),
                  '1011': complex(3 * d, -3 * d),
                  '1100': complex(-d, -d),
                  '1101': complex(-d, -3 * d),
                  '1110': complex(-3 * d, -d),
                  '1111': complex(-3 * d, -3 * d)}

    # Make a lis of the constellation points for a 256-QAM with distance d between points
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

    # Channel estimation and equalization
    A = np.array([[11, 10], [10, 11]])
    B = np.kron(np.eye(np.size(received_signal) // 2), A)

    #Compute SVD of B
    U, S, V = np.linalg.svd(B, full_matrices=True)

    print("S", S)

    # Compute pseudo-inverse of B
    B_inv = np.diag(1 / S) @ U.T

    # Apply the pseudo-inverse of B to the received signal
    equalized_signal = B_inv @ received_signal

    print("Eq", equalized_signal)


    # 16-QAM demodulation separated by a distance of sqrt(24/15)
    d = 1
    symbol_map = {complex(d, d): '0000', complex(d, 3 * d): '0001', complex(3 * d, d): '0010',
                  complex(3 * d, 3 * d): '0011',
                  complex(-d, d): '0100', complex(-d, 3 * d): '0101', complex(-3 * d, d): '0110',
                  complex(-3 * d, 3 * d): '0111',
                  complex(d, -d): '1000', complex(d, -3 * d): '1001', complex(3 * d, -d): '1010',
                  complex(3 * d, -3 * d): '1011',
                  complex(-d, -d): '1100', complex(-d, -3 * d): '1101', complex(-3 * d, -d): '1110',
                  complex(-3 * d, -3 * d): '1111'}

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

    print("MAP", symbol_map)


    qam_signal = [complex(equalized_signal[i], equalized_signal[i + 1]) for i in range(0, len(equalized_signal), 2)]

    print("[RECEIVER] received signal QAM after equalizing:", qam_signal)


    demodulated_signal = [find_closest_point(point, symbol_map) for point in qam_signal]

    # Convert binary message back to text
    binary_message = ''.join(demodulated_signal)
    print("[RECEIVER] binary received signal demodulated:", binary_message)
    text_message = ''.join(chr(int(binary_message[i:i + 8], 2)) for i in range(0, len(binary_message), 8))

    # Return the first 50 characters of the text message
    return text_message[:50]


def channel(sent_signal):
    # print("[CHANNEL] sent signal SIZE:", len(sent_signal))
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
    # print("[CHANNEL] noised signal SIZE:", len(Y))
    return Y


def example_usage():
    def generate_random_string(length):
        # All ASCII characters
        ascii_characters = string.ascii_letters + string.digits + string.punctuation
        # Generate the random string
        random_string = ''.join(random.choice(ascii_characters) for _ in range(length))
        return random_string

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
    qpsk_signal = transmitter(message)

    # Save the signal to the file
    with open(filename, 'w') as f:
        for sample in qpsk_signal:
            f.write(str(sample) + '\n')


def receive_message_from_file(filename):
    # Read the file
    with open(filename, 'r') as f:
        signal_parts = [float(line.strip()) for line in f]

    # Pair up the real and imaginary parts to recreate the complex symbols
    qpsk_signal = [signal_parts for i in range(0, len(signal_parts), 2)]
    print("[READER] Signal:", qpsk_signal)
    # Pass the symbols to the receiver function
    message = receiver(qpsk_signal)

    return message


if __name__ == '__main__':
    # message = "Hello World"
    # save_transmitted_signal_to_file(message, "input.txt")
    #
    # save_transmitted_signal_to_file(message, "input.txt")
    #
    # receive_message_from_file("output.txt")

    # message = "Hello World!!"
    # save_transmitted_signal_to_file(message, "input.txt")

    # print(receive_message_from_file("output.txt"))


    total_errors = 0
    for x in range(100):
        total_errors += example_usage()
        print("")
    print("Total messages error:", total_errors)