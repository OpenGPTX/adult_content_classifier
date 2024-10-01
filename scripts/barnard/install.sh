#!/bin/bash

load_modules(){
    source /software/foundation/generic/10_modules.sh


    # Load the required modules
    # Update the version of the modules if needed
    module --force purge

    module load release/24.04
    module load GCCcore/13.2.0
    module load Python/3.11.5
    module load poetry/1.6.1

    module load slurm/slurm-paths

}
set_ws_poetry(){

    # Set the workspace name
    YOUR_NAME=$(whoami)
    VENV_NAME=pipeline_commons
    VENV_DIR=/data/horse/ws/$YOUR_NAME-$VENV_NAME


    # Allocate workspace (only if needed)
    ws_allocate $VENV_NAME 1

    # Find poetry config file
    POETRY_CONFIG_FILE="$HOME/.config/pypoetry/config.toml"

    # Check if the config file exists
    if [ ! -f "$POETRY_CONFIG_FILE" ]; then
        # Create the config file
        mkdir -p $HOME/.config/pypoetry
        touch $POETRY_CONFIG_FILE
    fi

    # Check if the config file is empty
    if [ ! -s "$POETRY_CONFIG_FILE" ]; then
        # Append the following lines to the config file
        echo "[virtualenvs]" >> $POETRY_CONFIG_FILE
        #echo "in-project = true" >> $POETRY_CONFIG_FILE
        echo "create = true" >> $POETRY_CONFIG_FILE
        echo "path = \"$VENV_DIR\"" >> $POETRY_CONFIG_FILE
    fi

     #  Set the global virtualenvs.path to the workspace directory
    poetry config virtualenvs.path $VENV_DIR
}

authenticate(){
    # Authenticate with GitLab (this will prompt you to enter your token)
    poetry config http-basic.pipelines $(whoami)
}

firts_time_install(){
    load_modules
    set_ws_poetry
    poetry install -vvv
}

refresh_install(){
    load_modules
    #poetry update -vvv
    # Install dependencies with Poetry
    poetry install -vvv
}
