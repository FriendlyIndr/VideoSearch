What does it do?

VideoSearch enables users to search within their own database of indexed YouTube videos and explanations associated with them.

How does it work?

User enters a link into the ingestion section. The system fetches the YouTube transcripts and stores them in the database as embeddings.
The user can then add their own explanations to the indexed videos. These user-added explanations will also be considered during a search operation.

The embeddings are currently in a pickle file. This reduces the need to run a vector DB in the background.

Limitations:

- The pickle storage currently being used to store vector embeddings has to be loaded completely into the RAM
  to be used, which does not scale for large datasets.
- Pickle also does not index vectors. So, to find similar vectors, obne must run a linear search, which is slow.
- Pickle also should not be used with untrusted input, as it can create a security vulnerability.

Tech Stack:

- Backend: FastAPI
- Frontend: HTML, CSS, JavaScript
