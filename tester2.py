import numpy as np
import string
import random


def transmitter(message):
    # Convert text message to binary representation
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    print("[TRANSMITTER] binary message:", binary_message)

    # Group bits into 4-bit chunks for 16-QAM
    binary_chunks = [binary_message[i:i + 4] for i in range(0, len(binary_message), 4)]

    # 16-QAM modulation
    symbol_map = {'0000': complex(1, 1), '0001': complex(1, 3), '0010': complex(3, 1), '0011': complex(3, 3),
                  '0100': complex(-1, 1), '0101': complex(-1, 3), '0110': complex(-3, 1), '0111': complex(-3, 3),
                  '1000': complex(1, -1), '1001': complex(1, -3), '1010': complex(3, -1), '1011': complex(3, -3),
                  '1100': complex(-1, -1), '1101': complex(-1, -3), '1110': complex(-3, -1), '1111': complex(-3, -3)}
    qam_signal = [symbol_map[chunk] for chunk in binary_chunks]

    # Separate the real and imaginary parts of the QAM signal
    signal_parts = [(sample.real, sample.imag) for sample in qam_signal]

    # Flatten the list of tuples into a single list
    flat_signal = [part for sample in signal_parts for part in sample]

    print("[TRANSMITTER] QAM signal:", flat_signal)
    return flat_signal


def receiver(received_signal):
    def find_closest_point(point, symbol_map):
        # Find the constellation point closest to the received point
        distances = [abs(point - constellation_point) for constellation_point in symbol_map.keys()]
        closest_point = min(distances)
        return symbol_map[list(symbol_map.keys())[distances.index(closest_point)]]

    # We take only the first sublist ?? Why not other ??
    received_signal = np.array(received_signal)[0]
    print("[RECEIVER] received signal:", received_signal)

    qam_signal = [complex(received_signal[i], received_signal[i + 1]) for i in range(0, len(received_signal), 2)]

    print("[RECEIVER] received signal QAM:", qam_signal)

    # 16-QAM demodulation
    symbol_map = {complex(1, 1): '0000', complex(1, 3): '0001', complex(3, 1): '0010', complex(3, 3): '0011',
                  complex(-1, 1): '0100', complex(-1, 3): '0101', complex(-3, 1): '0110', complex(-3, 3): '0111',
                  complex(1, -1): '1000', complex(1, -3): '1001', complex(3, -1): '1010', complex(3, -3): '1011',
                  complex(-1, -1): '1100', complex(-1, -3): '1101', complex(-3, -1): '1110', complex(-3, -3): '1111'}

    demodulated_signal = [find_closest_point(point, symbol_map) for point in qam_signal]

    # Convert binary message back to text
    binary_message = ''.join(demodulated_signal)
    print("[RECEIVER] binary received signal demodulated:", binary_message)
    text_message = ''.join(chr(int(binary_message[i:i + 8], 2)) for i in range(0, len(binary_message), 8))

    return text_message


def channel(sent_signal):
    sent_signal = np.array(sent_signal)
    assert np.size(sent_signal) <= 200, "n must be <= 100"
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


def example_usage():
    def generate_random_string(length):
        # All ASCII characters
        ascii_characters = string.ascii_letters + string.digits + string.punctuation
        # Generate the random string
        random_string = ''.join(random.choice(ascii_characters) for _ in range(length))
        return random_string

    # Example usage:
    message = generate_random_string(50)
    message = "hello"

    X = transmitter(message)  # Encode our message
    Y = channel(X)  # Simulate the treatment done by the channel
    reconstructed_message = receiver(Y)  # Decode the message received by the channel

    print("Original message:", message)
    print("Reconstructed message:", reconstructed_message)

    return message != reconstructed_message


def save_transmitted_signal_to_file(message, filename):
    # Transmit the message
    qpsk_signal = transmitter(message)

    # Separate the real and imaginary parts of the QPSK signal
    signal_parts = [(sample.real, sample.imag) for sample in qpsk_signal]

    # Flatten the list of tuples into a single list
    flat_signal = [part for sample in signal_parts for part in sample]

    # Save the signal to the file
    with open(filename, 'w') as f:
        for sample in flat_signal:
            f.write(str(sample) + '\n')


def receive_message_from_file(filename):
    # Read the file
    with open(filename, 'r') as f:
        signal_parts = [float(line.strip()) for line in f]

    # Pair up the real and imaginary parts to recreate the complex symbols
    qpsk_signal = [complex(signal_parts[i], signal_parts[i + 1]) for i in range(0, len(signal_parts), 2)]

    # Pass the symbols to the receiver function
    message = receiver(qpsk_signal)

    return message


if __name__ == '__main__':
    # message = "Hello World"
    # message = transmitter(message)
    #
    # save_transmitted_signal_to_file(message, "input.txt")
    #
    # receive_message_from_file("output.txt")


    total_errors = 0
    for x in range(1):
        total_errors += example_usage()
        print("")
    print("Total messages error:", total_errors)
