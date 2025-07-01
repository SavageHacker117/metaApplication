export class GameContainer {
    /**
     * @param {string} [id=\'game-container\'] - The ID for the game container div.
     */
    constructor(id = 'game-container') {
        this.id = id;
        this.containerElement = this._createContainer();
        console.log(`GameContainer initialized with ID: ${this.id}`);
    }

    _createContainer() {
        let container = document.getElementById(this.id);
        if (!container) {
            container = document.createElement('div');
            container.id = this.id;
            document.body.appendChild(container);
        }
        // Apply basic styles to make it full screen
        container.style.position = 'fixed';
        container.style.top = '0';
        container.style.left = '0';
        container.style.width = '100%';
        container.style.height = '100%';
        container.style.overflow = 'hidden';
        container.style.margin = '0';
        container.style.padding = '0';
        container.style.zIndex = '0'; // Ensure it's behind loading screen initially
        return container;
    }

    /**
     * Returns the HTML element of the game container.
     * @returns {HTMLElement}
     */
    getElement() {
        return this.containerElement;
    }

    /**
     * Sets the visibility of the game container.
     * @param {boolean} visible - True to show, false to hide.
     */
    setVisible(visible) {
        this.containerElement.style.display = visible ? 'block' : 'none';
    }
}


