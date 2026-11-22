-- The code index a render from a head made a derived version's link under.
--
-- `version_artifact_head.code_index_stamp` is the content hash of the code index
-- (`CodeIndex.stamp.content_hash`) the last write naming that head rendered the link's file
-- under. A sum's levels, a party-totals cut and the summable axes follow the code index, so a
-- render from one head under another index can make another version. An analysis root reads
-- the version a render from its dataset version recorded in place of rendering it again only
-- while a link of its minting names that dataset version under the serving index's hash
-- (`datasets_api.analysis_roots.LiveRootSource._recorded_rendering`).
-- `datasets_api.version_store._insert_artifact` writes it on every head row a derived write
-- names, replacing the hash an earlier write left. NULL on a row written before this column or
-- by the previous image, which the lookup treats as recorded under no index, so the first read
-- after this migration renders once and its write sets the hash. The frame refresh does not
-- read it.
--
-- Adding a nullable column rewrites no row. ALTER TABLE takes ACCESS EXCLUSIVE on
-- `version_artifact_head`, and `lock_timeout` fails it after five seconds so that it does not
-- queue every later write behind it. The previous image inserts head rows without it and leaves
-- an existing row as it is. Re-running this migration changes nothing.

SET LOCAL lock_timeout = '5s';

ALTER TABLE datasets.version_artifact_head
    ADD COLUMN IF NOT EXISTS code_index_stamp text;
