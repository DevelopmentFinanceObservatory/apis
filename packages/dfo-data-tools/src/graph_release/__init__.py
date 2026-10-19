"""The graph release: one build of the Data Commons graph and the DFO layer
per import, published to a bucket and loaded into memory by each serving
process.

`format` defines the bucket layout, the part schemas, the manifest and the
reader; nothing else names a column, a key or a format number.
"""
