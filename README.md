<p align="center">
  <a href="./"><img src="./docs/assets/grug.png" alt="Grug" width="200"></a>
</p>

# Grug Discord Agent

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/mypy-checked-blue)](https://github.com/python/mypy)

Grug is a self-hostable tabletop-RPG GenAI Agent designed to enhance your RPG experience by providing intelligent
responses and interactions.

## Features

- **Text Responses**: Can respond in discord chats when users @mention the agent, or when DM'd.
- **Listens In Voice Chat**: Listens to voice chat and responds intelligently.
- **AoN Integration**: Can look up rules and information from the Archives of Nethys (AoN) for Pathfinder.
- **Dice Roller**: Can roll dice and provide results in chat.
- **Image Generation**: Can generate images based on text prompts.

## Adding Grug to Your Server

TODO: instructions for how to sign up and use the currently deployed Grug (not ready yet)

## [Self-Hosting Grug](docs/self_hosting.md)

## Local Development

We welcome contributions! If you would like to contribute to Grug, or just want to run it locally, you can do so by
following these steps:

### Prerequisites

- [Install uv](https://docs.astral.sh/uv/getting-started/installation/#installing-uv) for package management and python
  version/environment management.
- [Install Docker Desktop](https://www.docker.com/products/docker-desktop/) for containerization.
  > NOTE: while docker desktop is free for personal use, if you'd rather only
  > [install docker](https://docs.docker.com/get-started/get-docker/) itself you can do that too. Note that you will
  > also need [docker-compose](https://docs.docker.com/compose/install/) to run everything in this project.
- [Install Git](https://git-scm.com/)

### Installation

1. Clone the repository:
   ```bash
   git clone
   ```
2. Change into the directory:
   ```bash
   cd grug
   ```
3. setup your python environment:
   ```bash
   uv sync
   ```
4. Start the Postgres database:
   ```bash
   docker-compose up -d postgres
   ```
   > NOTE: this will expose postgres on port 5432 by default. The default username and password are `postgres` and
   > `postgres`. you can adjust all of this with ENV variables.

5. Set up your configuration by following the instructions in the `./config/secrets.template.env` file.
   > NOTE: you will need a Discord bot token and an OpenAI API key to run Grug. You can get a Discord bot token by
   > creating a new bot application in the [Discord Developer Portal](https://discord.com/developers/applications)
   > and an OpenAI API from the [OpenAI Dashboard](https://platform.openai.com/api-keys). Everything with discord is
   > free, OpenAI costs can vary depending on usage and which model you choose to use. I have been using the
   > GPT-4o-mini and spend less than $5 a month on it for basic personal use, but YMMV.

6. Run the tests:
   ```bash
   uv run test
   ```
7. Run the application:
   ```bash
   uv run grug
   ```
8. TODO: create a config.env file for simple adjustments users can make before diving into changing any of the code

### Running with Docker

To start the entire application locally with docker run:

```bash
docker compose up -d
```

> this will also start the f5tts server which will expose a web server accessible at `http://localhost:7860` where you
> can test out the TTS capabilities. It is dependent on having an NVIDIA GPU and your system being configured to
> allow Docker to have access to the GPU, which typically requires extra setup. If this raises errors, just start the
> application service with `docker compose up -d application` and it will run without the TTS server.
