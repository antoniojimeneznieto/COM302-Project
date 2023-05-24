import numpy as np


# Simple and incomplete sketch:
def text_to_bits(text):
    bits = bin(int.from_bytes(text.encode(), 'big'))[2:]
    return list(map(int, bits.zfill(8 * ((len(bits) + 7) // 8))))


def bits_to_text(bits):
    n = int(''.join(map(str, bits)), 2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode()


def transmitter(text):  # TODO: Add error-detection and coding and modulation
    bits = text_to_bits(text)
    samples = [1 if bit == 1 else -1 for bit in bits]
    return np.array(samples)


def receiver(y):  # TODO: Add decoding and demodulation
    bits = [1 if sample >= 0 else 0 for sample in y]
    text = bits_to_text(bits)
    return text


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


if __name__ == '__main__':
    # Example usage:
    text = 'Hello, World!'
    X = transmitter(text)  # Encode our message
    Y = channel(X)  # Simulate the treatment done by the channel
    received_text = receiver(Y)  # Decode the message received by the channel

    print(received_text)  # It should decode the original text
