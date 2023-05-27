import numpy as np
import string
import random


def transmitter(message):
    # Convert text message to binary representation
    binary_message = ''.join(format(ord(char), '08b') for char in message)

    # Group bits into 2-bit chunks for QPSK
    binary_chunks = [binary_message[i:i + 2] for i in range(0, len(binary_message), 2)]

    # QPSK modulation
    symbol_map = {'00': complex(1, 1), '01': complex(-1, 1), '11': complex(-1, -1), '10': complex(1, -1)}
    qpsk_signal = [symbol_map[chunk] for chunk in binary_chunks]

    return qpsk_signal


def receiver(received_signal):
    # Flatten the received signal and convert to a list
    received_signal = np.array(received_signal).flatten().tolist()

    # QPSK demodulation
    symbol_map = {complex(1, 1): '00', complex(-1, 1): '01', complex(-1, -1): '11', complex(1, -1): '10'}
    demodulated_signal = [find_closest_point(point, symbol_map) for point in received_signal]

    # Convert binary message back to text
    binary_message = ''.join(demodulated_signal)
    text_message = ''.join(chr(int(binary_message[i:i + 8], 2)) for i in range(0, len(binary_message), 8))

    return text_message


def find_closest_point(point, symbol_map):
    # Find the constellation point closest to the received point
    distances = [abs(point - constellation_point) for constellation_point in symbol_map.keys()]
    closest_point = min(distances)
    return symbol_map[list(symbol_map.keys())[distances.index(closest_point)]]


def channel(sent_signal):
    sent_signal = np.array(sent_signal)
    assert np.size(sent_signal) <= 200, "n must be <= 100"  # TODO: We must reduce N from 350 to at least 100
    n = np.size(sent_signal) // 2
    x = sent_signal[0:2 * n]
    s = sum(x ** 2) / np.size(x)
    sigma = 1
    if s > 1:
        sigma = np.sqrt(s)
    Z = np.random.normal(0, sigma, size=(2 * n, 1))
    A = np.array([[11, 10], [10, 11]])
    B = np.kron(np.eye(n), A)
    Y = B.dot(x) + Z
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

    pos = reconstructed_message.find(message[0])
    reconstructed_message = reconstructed_message[pos: pos + len(message)]

    print("Original message:", message)
    print("Reconstructed message:", reconstructed_message)

    return message != reconstructed_message


if __name__ == '__main__':
    total_errors = 0
    for x in range(1):
        total_errors += example_usage()
        print("")
    print("Total messages error:", total_errors)
