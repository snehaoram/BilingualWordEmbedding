# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import embeddings
from cupy_utils import *

import argparse
import collections
import numpy as np
import sys


BATCH_SIZE = 500



def bilingual_induction(src):
    # Parse command line arguments
    srcfile_path = "SRC_MAPPED.emb"
    trgfile_path = "TRG_MAPPED.emb"
    parser = argparse.ArgumentParser(description='Evaluate embeddings of two languages in a shared space in word translation induction')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--seed', type=int, default=0, help='the random seed')
    args = parser.parse_args()

    # Choose the right dtype for the desired precision

    dtype = 'float32'


    # Read input embeddings
    srcfile =  open(srcfile_path, encoding=args.encoding, errors='surrogateescape')
    trgfile =  open(trgfile_path, encoding=args.encoding, errors='surrogateescape')
    src_words, x = embeddings.read(srcfile, dtype=dtype)
    trg_words, z = embeddings.read(trgfile, dtype=dtype)

    xp = np
    xp.random.seed(args.seed)

    # Length normalize embeddings so their dot product effectively computes the cosine similarity
    embeddings.length_normalize(x)
    embeddings.length_normalize(z)

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    src_ind = src_word2ind[src]
    # trg_ind = trg_word2ind[trg]
    punjabi = src
    src = src_ind

    translation1 = collections.defaultdict(int)

    similarities = x[src].dot(z.T)

    index_array = np.argpartition(similarities, kth=278552, axis=-1)
    b = np.sort(index_array[-5:])
    translation1[src] = b
    # a = list(trg_word2ind.keys())[list(trg_word2ind.values()).index(trg_ind)]
    for j in range(5):
      r = list(trg_word2ind.keys())[list(trg_word2ind.values()).index(translation1[src_ind][j])]
      print(str(j+1) + "." + r)
    




import streamlit as st

st.title('Bilingual Lexical Induction - Punjabi to Tamil')

word = st.chat_input("Punjabi Word")
if word:
    st.write(f"Input: {word}")
    bilingual_induction(word)
    
        