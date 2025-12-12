# tokenizer.py
class SimpleTokenizer:
    def __init__(self, vocab_size=None):
        self.vocab_size = 128  # ASCII 0-127

    def encode(self, text):
        return [min(ord(c), 127) for c in text]

    def decode(self, ids):
        return ''.join([chr(i) for i in ids if 0 <= i < 128])