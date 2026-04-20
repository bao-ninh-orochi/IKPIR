# ikpir-server

Server-side state and protocol for Incremental-Keyword-PIR.

- `setup`    — build the SCF index over the database keys, derive the hint
               matrix, and serialize the params wire format.
- `respond`  — compute the server response for a client query in one round.
- `update`   — **Phase C contribution.** Incremental insert / delete / update
               that refreshes the preprocessing matrix without a full
               rebuild.

Populated across Phase B (`setup`, `respond`) and Phase C (`update`).
