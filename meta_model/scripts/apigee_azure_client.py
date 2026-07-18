#!/usr/bin/env python
"""Mayo Apigee Azure OpenAI chat-completions helper.

Configure tokens through the model YAML using one of:
- OAuth client credentials, recommended for long jobs:
  - oauth_client_id_env: name of an environment variable containing client_id
  - oauth_client_secret_env: name of an environment variable containing client_secret
  - oauth_token_url: token endpoint; defaults to https://mcc.apix.mayo.edu/oauth/token
- Static bearer token fallback:
  - api_key_env: name of an environment variable containing the bearer token
  - api_key_file: path to a text file containing the bearer token
  - api_key_file_env