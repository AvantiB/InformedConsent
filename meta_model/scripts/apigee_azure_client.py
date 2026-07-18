#!/usr/bin/env python
"""Small helper for Mayo Apigee Azure OpenAI chat-completions endpoint.

No tokens should be committed. Configure either:
- api_key_env: environment variable containing the bearer token; or
- api_key_file / api_key_file_env: text file containing the bearer token.

When api_key_file is used, the token is read on every request, so a long-running
job can continue after you overwrite the token file with a