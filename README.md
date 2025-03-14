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

## Documentation

- [Self-Hosting Grug](docs/self_hosting.md)

TODO: instructions for how to self-host

## Planned Features

- [ ] add docs for self hosting
- [ ] add docs for general usage and features of Grug
- [ ] rules lookup
    - users must upload their own rulebooks and content as that is typically closed source
    - some tools will be provided out of the box, such as AoN for Pathfinder, and
    - online resources that are open for use:
        - https://media.wizards.com/2018/dnd/downloads/DnD_BasicRules_2018.pdf
        - AoN for Pathfinder
- [x] dice roller
- [ ] music player
    - youtube search code: https://github.com/joetats/youtube_search/blob/master/youtube_search/__init__.py
    - youtube downloader: https://github.com/yt-dlp/yt-dlp
    - can use the above two to find and obtain music and then can create an agent to stream it into a voice channel
- [ ] session notes (by listening to the play session)
- [ ] scheduling and reminders
    - [ ] ability to send reminder for the upcoming session
    - [ ] food tracking feature (for in-person sessions where there is a rotation of who brings food)
    - [ ] ability to send reminder for who is bringing food
    - [ ] scheduling feature for when the next session will be, and who is available (find a time that works best for
      everyone)

## Local Development

### Documentation

We use MkDocs Material published to GitHub Pages for our documentation. We use [Mike]() to handle documentation
versioning which comes with a few different commands when serving the docs locally, i.e:

```bash
uv run mike serve
```

## References

- TTS:
    - [F5-TTS Hugging Face Space](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
    - [F5-TTS source code](https://github.com/SWivid/F5-TTS)
- STT:
    - https://github.com/Vaibhavs10/insanely-fast-whisper
    - https://github.com/systran/faster-whisper
- Self-Hostable AI Tools and Models:
    - https://technotim.live/posts/ai-stack-tutorial/
    - https://github.com/AUTOMATIC1111/stable-diffusion-webui
