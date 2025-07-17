"""
Run script for the Unified Crypto API
------------------------------------
This script starts the Unified Crypto API server.
"""

import uvicorn

if __name__ == "__main__":
    print("Starting Unified Crypto API server...")
    uvicorn.run("crypto_api.api:app", host="0.0.0.0", port=8000, reload=True)
    print("Server stopped.")