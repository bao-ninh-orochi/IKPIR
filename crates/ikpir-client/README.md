# ikpir-client

Client-side state and protocol for Incremental-Keyword-PIR.

- `setup`    — construct a `Client` from the server's hint matrix + params.
- `query`    — build a PIR query for a keyword; produces an ephemeral
               `QueryState`.
- `decrypt`  — decode the server's response into the requested record value.

Populated in Phase B.
