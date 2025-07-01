"""
Human-in-the-Loop Web Interface

This module provides a Flask-based web interface for collecting human feedback
on RL training episodes through a user-friendly web application.
"""

from flask import Flask, render_template_string, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import csv
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Configuration
FEEDBACK_FILE = 'web_feedback.csv'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

# HTML Templates
MAIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RL Training Feedback Interface</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .episode-info {
            background: #e8f4fd;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #2196F3;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        input[type="range"] {
            width: 100%;
            margin: 10px 0;
        }
        .rating-display {
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            color: #2196F3;
            margin-bottom: 10px;
        }
        textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            resize: vertical;
            font-family: inherit;
        }
        button {
            background: #2196F3;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
            margin-top: 10px;
        }
        button:hover {
            background: #1976D2;
        }
        .success {
            background: #4CAF50;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            display: none;
        }
        .error {
            background: #f44336;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            display: none;
        }
        .nav-buttons {
            display: flex;
            justify-content: space-between;
            margin-top: 20px;
        }
        .nav-buttons button {
            width: 45%;
        }
        .stats-section {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #eee;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #2196F3;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎮 RL Training Feedback Interface</h1>
        
        <div id="success-message" class="success"></div>
        <div id="error-message" class="error"></div>
        
        <div id="episode-section">
            <div class="episode-info">
                <h3>Episode Information</h3>
                <p><strong>Episode:</strong> <span id="episode-number">Loading...</span></p>
                <p><strong>Reward:</strong> <span id="episode-reward">Loading...</span></p>
                <p><strong>Steps:</strong> <span id="episode-steps">Loading...</span></p>
                <p><strong>Duration:</strong> <span id="episode-duration">Loading...</span></p>
            </div>
            
            <form id="feedback-form">
                <div class="form-group">
                    <label for="overall-rating">Overall Performance Rating</label>
                    <div class="rating-display" id="overall-display">5</div>
                    <input type="range" id="overall-rating" min="1" max="10" value="5" 
                           oninput="updateRatingDisplay('overall-rating', 'overall-display')">
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #666;">
                        <span>Poor (1)</span>
                        <span>Excellent (10)</span>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="strategy-rating">Strategy Rating</label>
                    <div class="rating-display" id="strategy-display">5</div>
                    <input type="range" id="strategy-rating" min="1" max="10" value="5"
                           oninput="updateRatingDisplay('strategy-rating', 'strategy-display')">
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #666;">
                        <span>Poor Strategy (1)</span>
                        <span>Excellent Strategy (10)</span>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="efficiency-rating">Efficiency Rating</label>
                    <div class="rating-display" id="efficiency-display">5</div>
                    <input type="range" id="efficiency-rating" min="1" max="10" value="5"
                           oninput="updateRatingDisplay('efficiency-rating', 'efficiency-display')">
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #666;">
                        <span>Inefficient (1)</span>
                        <span>Very Efficient (10)</span>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="comments">Comments (Optional)</label>
                    <textarea id="comments" rows="4" 
                              placeholder="Any additional observations or suggestions..."></textarea>
                </div>
                
                <button type="submit">Submit Feedback</button>
            </form>
            
            <div class="nav-buttons">
                <button onclick="loadPreviousEpisode()">← Previous Episode</button>
                <button onclick="loadNextEpisode()">Next Episode →</button>
            </div>
        </div>
        
        <div class="stats-section">
            <h3>Feedback Statistics</h3>
            <div class="stats-grid" id="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="total-feedback">0</div>
                    <div class="stat-label">Total Feedback</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="avg-rating">0.0</div>
                    <div class="stat-label">Average Rating</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="latest-episode">0</div>
                    <div class="stat-label">Latest Episode</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let currentEpisode = 1;
        let episodeData = {};
        
        function updateRatingDisplay(sliderId, displayId) {
            const slider = document.getElementById(sliderId);
            const display = document.getElementById(displayId);
            display.textContent = slider.value;
        }
        
        function showMessage(message, isError = false) {
            const successDiv = document.getElementById('success-message');
            const errorDiv = document.getElementById('error-message');
            
            if (isError) {
                errorDiv.textContent = message;
                errorDiv.style.display = 'block';
                successDiv.style.display = 'none';
            } else {
                successDiv.textContent = message;
                successDiv.style.display = 'block';
                errorDiv.style.display = 'none';
            }
            
            setTimeout(() => {
                successDiv.style.display = 'none';
                errorDiv.style.display = 'none';
            }, 3000);
        }
        
        function loadEpisodeData(episodeNum) {
            fetch(`/api/episode/${episodeNum}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        showMessage(data.error, true);
                        return;
                    }
                    
                    episodeData = data;
                    currentEpisode = episodeNum;
                    
                    document.getElementById('episode-number').textContent = data.episode;
                    document.getElementById('episode-reward').textContent = data.reward.toFixed(2);
                    document.getElementById('episode-steps').textContent = data.steps;
                    document.getElementById('episode-duration').textContent = data.duration || 'N/A';
                })
                .catch(error => {
                    showMessage('Failed to load episode data', true);
                    console.error(error);
                });
        }
        
        function loadNextEpisode() {
            loadEpisodeData(currentEpisode + 1);
        }
        
        function loadPreviousEpisode() {
            if (currentEpisode > 1) {
                loadEpisodeData(currentEpisode - 1);
            }
        }
        
        function loadStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('total-feedback').textContent = data.total_feedback;
                    document.getElementById('avg-rating').textContent = data.avg_rating.toFixed(1);
                    document.getElementById('latest-episode').textContent = data.latest_episode;
                })
                .catch(error => console.error('Failed to load stats:', error));
        }
        
        document.getElementById('feedback-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const feedbackData = {
                episode: currentEpisode,
                reward: episodeData.reward,
                steps: episodeData.steps,
                overall_rating: parseInt(document.getElementById('overall-rating').value),
                strategy_rating: parseInt(document.getElementById('strategy-rating').value),
                efficiency_rating: parseInt(document.getElementById('efficiency-rating').value),
                comments: document.getElementById('comments').value
            };
            
            fetch('/api/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(feedbackData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showMessage('Feedback submitted successfully!');
                    loadStats();
                    // Reset form
                    document.getElementById('comments').value = '';
                    // Load next episode
                    setTimeout(() => loadNextEpisode(), 1000);
                } else {
                    showMessage(data.error || 'Failed to submit feedback', true);
                }
            })
            .catch(error => {
                showMessage('Failed to submit feedback', true);
                console.error(error);
            });
        });
        
        // Initialize
        loadEpisodeData(1);
        loadStats();
        setInterval(loadStats, 30000); // Update stats every 30 seconds
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Serve the main feedback interface."""
    return render_template_string(MAIN_TEMPLATE)


@app.route('/api/episode/<int:episode_num>')
def get_episode(episode_num):
    """Get episode data for feedback."""
    try:
        # Generate sample episode data (in real implementation, this would come from training logs)
        import random
        
        episode_data = {
            'episode': episode_num,
            'reward': round(random.uniform(-10, 50), 2),
            'steps': random.randint(50, 500),
            'duration': f"{random.randint(30, 300)}s",
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(episode_data)
    
    except Exception as e:
        logger.error(f"Error getting episode {episode_num}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback for an episode."""
    try:
        feedback_data = request.get_json()
        
        # Add timestamp
        feedback_data['timestamp'] = datetime.now().isoformat()
        
        # Save to CSV
        save_feedback_to_csv(feedback_data)
        
        logger.info(f"Feedback submitted for episode {feedback_data.get('episode')}")
        return jsonify({'success': True})
    
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def get_stats():
    """Get feedback statistics."""
    try:
        feedback_data = load_feedback_from_csv()
        
        if not feedback_data:
            return jsonify({
                'total_feedback': 0,
                'avg_rating': 0.0,
                'latest_episode': 0
            })
        
        # Calculate statistics
        total_feedback = len(feedback_data)
        ratings = [f.get('overall_rating', 0) for f in feedback_data if f.get('overall_rating')]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
        latest_episode = max([f.get('episode', 0) for f in feedback_data])
        
        return jsonify({
            'total_feedback': total_feedback,
            'avg_rating': avg_rating,
            'latest_episode': latest_episode
        })
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export')
def export_feedback():
    """Export feedback data as JSON."""
    try:
        feedback_data = load_feedback_from_csv()
        return jsonify(feedback_data)
    
    except Exception as e:
        logger.error(f"Error exporting feedback: {e}")
        return jsonify({'error': str(e)}), 500


def save_feedback_to_csv(feedback_data: Dict[str, Any]):
    """Save feedback data to CSV file."""
    filepath = Path(FEEDBACK_FILE)
    
    # Check if file exists to determine if we need headers
    file_exists = filepath.exists()
    
    with open(filepath, 'a', newline='') as csvfile:
        fieldnames = [
            'timestamp', 'episode', 'reward', 'steps', 
            'overall_rating', 'strategy_rating', 'efficiency_rating', 
            'comments'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(feedback_data)


def load_feedback_from_csv() -> List[Dict[str, Any]]:
    """Load feedback data from CSV file."""
    filepath = Path(FEEDBACK_FILE)
    
    if not filepath.exists():
        return []
    
    feedback_data = []
    try:
        with open(filepath, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Convert numeric fields
                for field in ['episode', 'steps', 'overall_rating', 'strategy_rating', 'efficiency_rating']:
                    if row.get(field):
                        try:
                            row[field] = int(row[field])
                        except ValueError:
                            pass
                
                if row.get('reward'):
                    try:
                        row['reward'] = float(row['reward'])
                    except ValueError:
                        pass
                
                feedback_data.append(row)
    
    except Exception as e:
        logger.error(f"Error loading feedback CSV: {e}")
    
    return feedback_data


def create_app(config: Dict[str, Any] = None) -> Flask:
    """Create and configure Flask app."""
    if config:
        app.config.update(config)
    
    # Setup logging
    if not app.debug:
        logging.basicConfig(level=logging.INFO)
    
    return app


if __name__ == '__main__':
    # Development server
    app.run(host='0.0.0.0', port=5000, debug=True)

