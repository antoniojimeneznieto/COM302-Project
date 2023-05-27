import numpy as np
import string
import random


# Simple and incomplete sketch:
def transmitter(message):
    # Convert text message to binary representation
    binary_message = ''.join(format(ord(char), '07b') for char in message)

    # BPSK modulation
    binary_signal = [int(bit) * 2 - 1 for bit in binary_message]

    return binary_signal


def receiver(received_signal):
    # Channel parameters
    n = len(received_signal) // 2
    x = np.array(received_signal[0:2 * n])

    # Reshape x to be one-dimensional
    x = x.flatten()

    # BPSK demodulation
    demodulated_signal = ['1' if bit > 0 else '0' for bit in x]

    # Convert binary message back to text
    binary_message = ''.join(demodulated_signal)

    # Remove padding zeros
    padded_zeros = len(binary_message) % 7
    if padded_zeros != 0:
        binary_message = binary_message[:-padded_zeros]

    text_message = ''.join(chr(int(binary_message[i:i + 7], 2)) for i in range(0, len(binary_message), 7))

    return text_message


def channel(sent_signal):
    sent_signal = np.array(sent_signal)
    assert np.size(sent_signal) <= 350, "n must be <= 100"  # TODO: We must reduce N from 350 to at least 100
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
    for x in range(100):
        total_errors += example_usage()
        print("")
    print("Total messages error:", total_errors)
