/**
 * DARK MATTER - Tower Defense UI Controller
 * Handles UI interactions and updates for the tower defense game
 */

class TowerDefenseUI {
    constructor() {
        this.gameInstance = null;
        this.elements = {};
        this.isInitialized = false;
        
        this.init();
    }
    
    init() {
        this.cacheElements();
        this.setupEventListeners();
        this.isInitialized = true;
        
        console.log('[DARK MATTER] Tower Defense UI initialized');
    }
    
    cacheElements() {
        // Stats elements
        this.elements.livesValue = document.getElementById('lives-value');
        this.elements.goldValue = document.getElementById('gold-value');
        this.elements.scoreValue = document.getElementById('score-value');
        this.elements.waveValue = document.getElementById('wave-value');
        
        // Control buttons
        this.elements.startWaveBtn = document.getElementById('start-wave-btn');
        this.elements.pauseBtn = document.getElementById('pause-btn');
        this.elements.resetBtn = document.getElementById('reset-btn');
        this.elements.spawnEnemyBtn = document.getElementById('spawn-enemy-btn');
        
        // Wave progress elements
        this.elements.waveProgressBar = document.getElementById('wave-progress-bar');
        this.elements.enemiesRemaining = document.getElementById('enemies-remaining');
        this.elements.waveStatus = document.getElementById('wave-status');
        
        // Notification elements
        this.elements.gameOverNotification = document.getElementById('game-over-notification');
        this.elements.waveCompleteNotification = document.getElementById('wave-complete-notification');
        this.elements.finalScore = document.getElementById('final-score');
        
        // Tower info tooltip
        this.elements.towerInfo = document.getElementById('tower-info');
        this.elements.towerRange = document.getElementById('tower-range');
        this.elements.towerDamage = document.getElementById('tower-damage');
        this.elements.towerFirerate = document.getElementById('tower-firerate');
        this.elements.towerCost = document.getElementById('tower-cost');
    }
    
    setupEventListeners() {
        // Button event listeners
        this.elements.startWaveBtn.addEventListener('click', () => {
            this.startWave();
        });
        
        this.elements.pauseBtn.addEventListener('click', () => {
            this.pauseGame();
        });
        
        this.elements.resetBtn.addEventListener('click', () => {
            this.resetGame();
        });
        
        this.elements.spawnEnemyBtn.addEventListener('click', () => {
            this.spawnTestEnemy();
        });
        
        // Game state updates
        window.addEventListener('towerDefenseUpdate', (event) => {
            this.updateUI(event.detail);
        });
        
        // Game events
        window.addEventListener('gameOver', (event) => {
            this.showGameOver(event.detail.score);
        });
        
        window.addEventListener('waveComplete', (event) => {
            this.showWaveComplete();
        });
        
        // Mouse events for tower info tooltip
        document.addEventListener('mousemove', (event) => {
            this.updateTowerTooltip(event);
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (event) => {
            this.handleKeyboard(event);
        });
    }
    
    setGameInstance(gameInstance) {
        this.gameInstance = gameInstance;
        console.log('[DARK MATTER] UI connected to game instance');
    }
    
    updateUI(gameState) {
        if (!this.isInitialized) return;
        
        // Update stats
        this.elements.livesValue.textContent = gameState.lives;
        this.elements.goldValue.textContent = gameState.gold;
        this.elements.scoreValue.textContent = gameState.score;
        this.elements.waveValue.textContent = gameState.currentWave;
        
        // Update button states
        this.elements.startWaveBtn.disabled = gameState.isPlaying && !gameState.isPaused;
        this.elements.pauseBtn.textContent = gameState.isPaused ? 'Resume' : 'Pause';
        this.elements.pauseBtn.disabled = !gameState.isPlaying;
        
        // Update wave status
        if (gameState.isPlaying && !gameState.isPaused) {
            this.elements.waveStatus.textContent = 'In Progress';
            this.elements.waveStatus.style.color = '#ffff00';
        } else if (gameState.isPaused) {
            this.elements.waveStatus.textContent = 'Paused';
            this.elements.waveStatus.style.color = '#ff8800';
        } else {
            this.elements.waveStatus.textContent = 'Ready';
            this.elements.waveStatus.style.color = '#00ff00';
        }
        
        // Update enemies count (if game instance available)
        if (this.gameInstance) {
            this.elements.enemiesRemaining.textContent = this.gameInstance.enemies.length;
            
            // Update wave progress (simplified)
            const progress = gameState.currentWave > 0 ? 
                Math.min(100, (gameState.score / (gameState.currentWave * 100)) * 100) : 0;
            this.elements.waveProgressBar.style.width = progress + '%';
        }
        
        // Color coding for critical stats
        if (gameState.lives <= 5) {
            this.elements.livesValue.style.color = '#ff0000';
        } else if (gameState.lives <= 10) {
            this.elements.livesValue.style.color = '#ffff00';
        } else {
            this.elements.livesValue.style.color = '#ffffff';
        }
        
        if (gameState.gold < 50) {
            this.elements.goldValue.style.color = '#ff8800';
        } else {
            this.elements.goldValue.style.color = '#ffffff';
        }
    }
    
    startWave() {
        if (this.gameInstance) {
            this.gameInstance.startWave();
            this.showNotification('Wave Started!', 2000, '#00ff00');
        } else if (window.rubyEngine && window.rubyEngine.plugins.threejs_tower_defense) {
            window.rubyEngine.plugins.threejs_tower_defense.startWave();
        }
    }
    
    pauseGame() {
        if (this.gameInstance) {
            this.gameInstance.pauseGame();
        } else if (window.rubyEngine && window.rubyEngine.plugins.threejs_tower_defense) {
            window.rubyEngine.plugins.threejs_tower_defense.pauseGame();
        }
    }
    
    resetGame() {
        if (this.gameInstance) {
            this.gameInstance.resetGame();
            this.hideAllNotifications();
            this.showNotification('Game Reset!', 2000, '#00ff00');
        } else if (window.rubyEngine && window.rubyEngine.plugins.threejs_tower_defense) {
            window.rubyEngine.plugins.threejs_tower_defense.resetGame();
        }
    }
    
    spawnTestEnemy() {
        const enemyTypes = ['basic', 'fast', 'heavy', 'boss'];
        const randomType = enemyTypes[Math.floor(Math.random() * enemyTypes.length)];
        
        if (this.gameInstance) {
            this.gameInstance.spawnEnemy(randomType);
        } else if (window.rubyEngine && window.rubyEngine.plugins.threejs_tower_defense) {
            window.rubyEngine.plugins.threejs_tower_defense.spawnEnemy(randomType);
        }
    }
    
    updateTowerTooltip(event) {
        // Show tower info when hovering over placement areas
        const rect = document.getElementById('game-container').getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        // Simple logic to show tooltip in game area
        if (x > 200 && x < rect.width - 200 && y > 100 && y < rect.height - 100) {
            this.elements.towerInfo.style.left = (event.clientX + 10) + 'px';
            this.elements.towerInfo.style.top = (event.clientY - 10) + 'px';
            this.elements.towerInfo.style.display = 'block';
        } else {
            this.elements.towerInfo.style.display = 'none';
        }
    }
    
    handleKeyboard(event) {
        switch(event.code) {
            case 'Space':
                event.preventDefault();
                this.pauseGame();
                break;
            case 'Enter':
                event.preventDefault();
                this.startWave();
                break;
            case 'KeyR':
                if (event.ctrlKey) {
                    event.preventDefault();
                    this.resetGame();
                }
                break;
            case 'KeyE':
                event.preventDefault();
                this.spawnTestEnemy();
                break;
        }
    }
    
    showGameOver(score) {
        this.elements.finalScore.textContent = score;
        this.elements.gameOverNotification.style.display = 'block';
        
        // Auto-hide after 10 seconds
        setTimeout(() => {
            this.elements.gameOverNotification.style.display = 'none';
        }, 10000);
    }
    
    showWaveComplete() {
        this.elements.waveCompleteNotification.style.display = 'block';
        
        // Auto-hide after 3 seconds
        setTimeout(() => {
            this.elements.waveCompleteNotification.style.display = 'none';
        }, 3000);
    }
    
    showNotification(message, duration = 3000, color = '#00ff00') {
        // Create temporary notification
        const notification = document.createElement('div');
        notification.className = 'notification';
        notification.style.display = 'block';
        notification.style.borderColor = color;
        notification.innerHTML = `<h3 style="color: ${color}; margin: 0;">${message}</h3>`;
        
        document.getElementById('ui-overlay').appendChild(notification);
        
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, duration);
    }
    
    hideAllNotifications() {
        this.elements.gameOverNotification.style.display = 'none';
        this.elements.waveCompleteNotification.style.display = 'none';
    }
    
    // Public API for external control
    updateStats(stats) {
        this.updateUI(stats);
    }
    
    showMessage(message, type = 'info') {
        const colors = {
            info: '#00ff00',
            warning: '#ffff00',
            error: '#ff0000',
            success: '#00ff88'
        };
        
        this.showNotification(message, 3000, colors[type] || colors.info);
    }
    
    setButtonState(buttonId, enabled) {
        const button = document.getElementById(buttonId);
        if (button) {
            button.disabled = !enabled;
        }
    }
    
    destroy() {
        // Clean up event listeners
        this.hideAllNotifications();
        console.log('[DARK MATTER] Tower Defense UI destroyed');
    }
}

// Initialize UI when DOM is loaded
let uiInstance = null;

document.addEventListener('DOMContentLoaded', () => {
    uiInstance = new TowerDefenseUI();
    
    // Make UI instance available globally for plugin integration
    window.towerDefenseUI = uiInstance;
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TowerDefenseUI;
}

// Global functions for HTML button onclick handlers
function startWave() {
    if (uiInstance) uiInstance.startWave();
}

function pauseGame() {
    if (uiInstance) uiInstance.pauseGame();
}

function resetGame() {
    if (uiInstance) uiInstance.resetGame();
}

function spawnTestEnemy() {
    if (uiInstance) uiInstance.spawnTestEnemy();
}

