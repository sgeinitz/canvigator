# Installation and setup

← [Back to main README](../README.md)

Note that installation and usage requires some basic knowledge on how
to use the command line. If necessary, there are many brief
tutorials/lessons available to help in this area, e.g.,
[freecodecamp.org](https://www.freecodecamp.org/news/command-line-for-beginners/).

1. **Clone the repository**:
    [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) this repository to your local system. This can either be done 
in just one place on your local system, or separately for each course that it will be used in. The latter option is our preferred method since we tend to have a separate directory for each course we teach, and prefer to keep the course data separate.

2. **Generate a Canvas API Token**:
    Naviagte to the `canvigator` directory then run configuration setup script by simply typing, `./configure.sh` at the terminal prompt.
   ```bash
   cd canvigator
   ```
   ```bash
   ./configure.sh
   ```
   You will be prompted to enter:
   - Your institution's Canvas LMS base URL (you can find this on your Canvas home page).
   - Your Canvas token, which can be created by navigating to _'Account'_, then _'Settings'_. Towards the bottom of this window in Canvas you will see a blue button, _'+ New Access Token_'. Click on this button to copy/download the token to your local system. DO NOT TO SHARE YOUR CANVAS TOKEN w/ ANYONE (e.g. do not save it in a shared directory).
   - Your Ollama API key (optional — only needed if you plan to use the cloud-hosted text model for the LLM-powered tasks below). Leave it blank if you plan to use only local Ollama models (or none at all). See the [Ollama setup](#ollama-setup-optional) section for how to generate one.

This will prompt the creation of the _data/_ and _figures/_ subdirectories.

3. **Verify Setup**:
    Once this is complete, double check that the configuration script, _set_env.sh_ has been created, that it has the correct values for your URL and token, and that the subdirectories have been created. 
    ```bash
    set_env.sh
    ```
    ```bash
    env
    ```
    This command will list all environment variables, allowing you to confirm that the necessary variables have been set correctly.

4. **Verify Python Libraries**:
    Before running the project, ensure that all required Python libraries are installed:
    1. Install the required libraries using `pip`:
    
    ```bash
    pip install -r requirements.txt 
    ```
    
    2. Alternatively, you can install each library individually if needed.

        * ![canvasapi](https://img.shields.io/badge/canvasapi-3.3.0-blue)
        * ![matplotlib](https://img.shields.io/badge/matplotlib-3.8.4-brightgreen)
        * ![numpy](https://img.shields.io/badge/numpy-1.26.4-yellow)
        * ![pandas](https://img.shields.io/badge/pandas-2.2.2-orange)
        * ![requests](https://img.shields.io/badge/requests-2.33.0-red)
        * ![scipy](https://img.shields.io/badge/scipy-1.13.1-lightgrey)
        * ![seaborn](https://img.shields.io/badge/seaborn-0.13.2-blueviolet)
        * ![ollama](https://img.shields.io/badge/ollama-0.4.7-black)

    If no errors are thrown, the libraries are successfully installed. The `ollama` client is only used by the LLM-assisted tasks described below — see [Ollama setup](#ollama-setup-optional).


## Ollama setup (optional)

Several tasks use a Large Language Model (LLM) via [Ollama](https://ollama.com) to draft and tag questions, generate open-ended follow-ups, transcribe student audio, and assess student replies. You can skip this section if you will not be running any of these tasks (`create-quiz` with the `[g]enerate w/ LLM` option, `get-quiz-questions --tag`, `generate-follow-up-questions`, `send-quiz-reminder`, `send-follow-up-question`, `assess-replies`).

Canvigator uses two kinds of models:

1. **A cloud-hosted text model** (default `gemini-3-flash-preview`, set via `OLLAMA_TEXT_MODEL`) for instructor-side text generation — drafting quiz questions in `create-quiz`, tagging quiz questions, and generating open-ended follow-up questions. These tasks never see student data, so a larger cloud model is a good fit.
2. **Local models** for tasks that process student input — `gemma4:31b` (default `OLLAMA_MODEL`) for assessing text/image replies, and `gemma4:e4b` (default `OLLAMA_AUDIO_MODEL`) for transcribing student audio. Keeping these local is deliberate: student submissions should not leave your machine.

**To use the cloud text model:**
1. Sign in at [ollama.com](https://ollama.com) and create an API key from your account settings.
2. Paste the key when prompted by `./configure.sh` (or add `export OLLAMA_API_KEY="..."` directly to `set_env.sh`).

**To use the local models:**
1. Install Ollama from [ollama.com/download](https://ollama.com/download) and start it (`ollama serve`, or use the Ollama desktop app).
2. Pull the models you need:
   ```bash
   ollama pull gemma4:31b   # assessment of text/image replies (assess-replies)
   ollama pull gemma4:e4b   # transcription of student audio (assess-replies, explain mode)
   ```
   Only pull the models for tasks you actually plan to run. `gemma4:31b` is a large download (~20 GB) and is only required for `assess-replies`.

All model names are overridable via env vars (`OLLAMA_TEXT_MODEL`, `OLLAMA_MODEL`, `OLLAMA_AUDIO_MODEL`), and the local host can be overridden via the standard `OLLAMA_HOST` env var.
