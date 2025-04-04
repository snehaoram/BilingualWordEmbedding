# Bilingual Word Embedding
- This is a course project on building a bilingual word embedding of two Indian languages with different families (tamil and punjabi), in CS626: Speech, Natural Language Processing and the Web, IIT Bombay.
- This work is implemented taking reference from the work in [1].
- At first monolingual word embedding of the individual languages are built.
- Next both the monoligual embeddings are made to come under a single embedding space with the help of linear transformation.
- In order to build the bilingual word embedding, a dictionary of only 100 word pairs is made of the two languages.
- Another task accomplished in this project is predicting the top-k of similar tamil words, given a word in punjabi and vice-versa.

  # Reference
  [1] Artetxe, M., Labaka, G., & Agirre, E. (2017, July). Learning bilingual word embeddings with (almost) no bilingual data. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 451-462). 
