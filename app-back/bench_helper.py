#!/usr/bin/env python3
"""Small helper used by smoke tests.

Reads a file path from argv[1] and prints the number of characters.
"""
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: bench_helper.py <input_file>')
        sys.exit(2)
    p = sys.argv[1]
    try:
        s = open(p, 'r', encoding='utf-8').read()
        print(len(s))
    except Exception as e:
        print('ERROR', e)
        sys.exit(1)
