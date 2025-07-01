
#!/bin/bash
# Script to manage and switch between V5 training configurations

CONFIG_DIR="/home/ubuntu/RLTrainingScriptforProceduralTowerDefenseGamev5beta1/v5beta1/config"

function show_help {
    echo "Usage: $0 [command]"
    echo "Commands:"
    echo "  list                     List available configuration files."
    echo "  use <config_file>        Set a configuration file as the default for demo_run.sh."
    echo "  create <new_config_name> Create a new configuration file from a template."
    echo "  edit <config_file>       Open a configuration file for editing."
    echo "  help                     Show this help message."
}

function list_configs {
    echo "Available V5 Configurations in $CONFIG_DIR:"
    ls -1 $CONFIG_DIR/*.json 2>/dev/null | xargs -n 1 basename
}

function use_config {
    local config_file=$1
    if [[ -f "$CONFIG_DIR/$config_file" ]]; then
        echo "Setting $config_file as the default configuration."
        # This assumes demo_run.sh looks for config_production.yaml by default
        # For this to work, demo_run.sh needs to be updated to use a symlink or similar
        # For now, we'll just inform the user.
        echo "Please remember to specify this config when running demo_run.sh:"
        echo "  bash demo_run.sh $config_file"
    else
        echo "Error: Configuration file '$config_file' not found in $CONFIG_DIR."
    fi
}

function create_config {
    local new_config_name=$1
    local template_config="$CONFIG_DIR/training_v5.json" # Assuming this is a good template
    local new_path="$CONFIG_DIR/$new_config_name.json"

    if [[ -f "$new_path" ]]; then
        echo "Error: Configuration file '$new_path' already exists."
    else
        cp "$template_config" "$new_path"
        echo "Created new configuration '$new_config_name.json' from template."
        echo "You can now edit it: $0 edit $new_config_name.json"
    fi
}

function edit_config {
    local config_file=$1
    local config_path="$CONFIG_DIR/$config_file"
    if [[ -f "$config_path" ]]; then
        echo "Opening '$config_file' for editing. Use your preferred editor (e.g., nano, vim)."
        echo "Once done, save and exit the editor."
        # This would ideally open a text editor, but in a sandboxed environment,
        # we'll just instruct the user.
        echo "To edit, you would typically run: nano $config_path or vim $config_path"
        echo "For now, you can use the 'file_read_text' and 'file_write_text' tools."
    else
        echo "Error: Configuration file '$config_file' not found in $CONFIG_DIR."
    fi
}

case "$1" in
    list)
        list_configs
        ;;
    use)
        use_config $2
        ;;
    create)
        create_config $2
        ;;
    edit)
        edit_config $2
        ;;
    help)
        show_help
        ;;
    *)
        show_help
        ;;
esac


