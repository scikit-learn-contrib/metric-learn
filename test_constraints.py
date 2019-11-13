from metric_learn.constraints import Constraints
partial_labels = [1, 2, 2, 1, -1, 3, 3]
cons = Constraints(partial_labels)
cons.chunks(num_chunks=3, chunk_size=2)
chunks = cons.chunks(num_chunks=3, chunk_size=2)
expected_chunk = [0, 1, 1, 0, -1, 2, 2]
print(chunks, expected_chunk)
