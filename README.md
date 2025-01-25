## Overview
The `smolvlm_api` project is designed to provide a lightweight and simple  OpenAI and Ollama-complaint API for handling various tasks. My goal was to make API for my own homeassistant + frigate bundle to integrate CV cameras with vision LLM that would be describe CV camera shots in a human-readable way. Service based on HuggingFace [SmolVLM-256M-Base](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Base/tree/main) model that could be user even on Raspberry Pi/Orange Pi consumer hardware.

## Features
- OpenAI and Ollama-complaint API
- Easy to integrate and use

## Usage Instructions
1. Clone the repository or just download the `docker compose.yml` file.
2. Run the following command to start the service: `docker compose up -d`.
3. The service will be available at `http://localhost:8000` by default.
4. That's it! You can now start using the API.

**Note**: First start, it will take some time to download the model and start the service (depends on your internet connection, need to download ~1.5GB model).

If you want to build the image yourself, you can run the following command: `docker compose up -d --build`.

## What else?

### Prompting

Default prompt is described in `src/config.py` file in `DEFAULT_PROMPT` constant. You can change it to your own needs. Another way is to pass the prompt as a query parameter in the request.

### API Endpoints and examples

FastAPI provides a nice interactive documentation for the API. You can access it at `http://localhost:8000/docs` or `http://localhost:8000/redoc`.

## Contributing
We welcome contributions from the community! If you would like to contribute to the project, please open an issue or submit a pull request.

## License
This project is licensed under the Apache 2.0. See the LICENSE file for more details.