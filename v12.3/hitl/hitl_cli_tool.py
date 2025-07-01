#!/usr/bin/env python3
"""
Standalone CLI Tool for HITL Feedback Collection - Version 5 BETA 1

This tool can be used independently to collect human feedback for training episodes.
It provides a clean, user-friendly command-line interface for rating episodes and
providing detailed feedback.

Usage:
    python hitl_cli_tool.py --episode 100 --reward 15.7 --steps 250
    python hitl_cli_tool.py --config feedback_config.json
    python hitl_cli_tool.py --interactive
"""

import argparse
import json
import csv
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HITLCLITool:
    """Standalone CLI tool for HITL feedback collection."""
    
    def __init__(self, csv_file: str = "hitl_feedback_cli.csv"):
        self.csv_file = Path(csv_file)
        self.rating_labels = {
            1: "Very Poor - Completely failed, no progress",
            2: "Poor - Major issues, minimal progress", 
            3: "Below Average - Some issues, limited progress",
            4: "Slightly Below Average - Minor issues, decent progress",
            5: "Average - Acceptable performance, room for improvement",
            6: "Slightly Above Average - Good performance, minor improvements needed",
            7: "Good - Strong performance, working well",
            8: "Very Good - Excellent performance, minor optimizations possible",
            9: "Excellent - Outstanding performance, very impressive",
            10: "Outstanding - Perfect performance, couldn't be better"
        }
        
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'episode', 'timestamp', 'rating', 'rating_label', 'comments', 
                    'suggestions', 'total_reward', 'steps', 'assets_used',
                    'performance_notes', 'improvement_areas', 'positive_aspects'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            
            logger.info(f"Initialized feedback CSV: {self.csv_file}")
    
    def collect_feedback(self, episode_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Collect comprehensive feedback for an episode."""
        
        print("\n" + "="*80)
        print("🎮 HUMAN-IN-THE-LOOP FEEDBACK COLLECTION - VERSION 5 BETA 1")
        print("="*80)
        
        # Display episode information
        self._display_episode_info(episode_data)
        
        try:
            # Collect rating
            rating = self._collect_rating()
            if rating is None:
                return None
            
            # Collect detailed feedback
            feedback_data = self._collect_detailed_feedback(episode_data, rating)
            
            # Save feedback
            self._save_feedback(feedback_data)
            
            # Display summary
            self._display_feedback_summary(feedback_data)
            
            return feedback_data
            
        except KeyboardInterrupt:
            print("\n\n❌ Feedback collection cancelled by user.")
            return None
        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
            return None
    
    def _display_episode_info(self, episode_data: Dict[str, Any]):
        """Display episode information in a formatted way."""
        print(f"📊 Episode: {episode_data.get('episode', 'Unknown')}")
        print(f"🏆 Total Reward: {episode_data.get('total_reward', 0):.2f}")
        print(f"👣 Steps Taken: {episode_data.get('steps', 0)}")
        
        assets = episode_data.get('assets_used', [])
        if assets:
            print(f"🎨 NeRF Assets Used: {', '.join(assets)}")
        else:
            print("🎨 NeRF Assets Used: None")
        
        performance = episode_data.get('performance_metrics', {})
        if performance:
            print("⚡ Performance Metrics:")
            for key, value in performance.items():
                print(f"   • {key}: {value}")
        
        scene_desc = episode_data.get('scene_description', '')
        if scene_desc:
            print(f"🎬 Scene: {scene_desc}")
        
        print("-" * 80)
    
    def _collect_rating(self) -> Optional[int]:
        """Collect rating with detailed scale explanation."""
        print("\n📝 Please rate this episode on a scale of 1-10:")
        print()
        
        # Display rating scale
        for rating, description in self.rating_labels.items():
            print(f"  {rating:2d}: {description}")
        
        print()
        print("💡 Consider: Strategy effectiveness, visual quality, NeRF asset usage,")
        print("    performance, creativity, and overall episode success.")
        print()
        
        while True:
            try:
                user_input = input("Enter rating (1-10) or 's' to skip: ").strip().lower()
                
                if user_input == 's':
                    print("⏭️  Feedback skipped.")
                    return None
                
                rating = int(user_input)
                if 1 <= rating <= 10:
                    print(f"✅ Rating selected: {rating} - {self.rating_labels[rating]}")
                    return rating
                else:
                    print("❌ Please enter a number between 1 and 10.")
                    
            except ValueError:
                print("❌ Please enter a valid number or 's' to skip.")
    
    def _collect_detailed_feedback(self, episode_data: Dict[str, Any], rating: int) -> Dict[str, Any]:
        """Collect detailed feedback beyond just rating."""
        
        print("\n" + "-" * 80)
        print("📋 DETAILED FEEDBACK COLLECTION")
        print("-" * 80)
        
        feedback_data = {
            'episode': episode_data.get('episode', 0),
            'timestamp': datetime.now().isoformat(),
            'rating': rating,
            'rating_label': self.rating_labels[rating],
            'total_reward': episode_data.get('total_reward', 0),
            'steps': episode_data.get('steps', 0),
            'assets_used': ', '.join(episode_data.get('assets_used', []))
        }
        
        # General comments
        print("\n💬 General Comments:")
        print("   What did you think about this episode overall?")
        comments = input("   > ").strip()
        feedback_data['comments'] = comments
        
        # Specific suggestions
        print("\n🔧 Suggestions for Improvement:")
        print("   What specific changes would improve performance?")
        suggestions = input("   > ").strip()
        feedback_data['suggestions'] = suggestions
        
        # Performance notes
        print("\n⚡ Performance Notes:")
        print("   Any observations about speed, rendering, or technical aspects?")
        performance_notes = input("   > ").strip()
        feedback_data['performance_notes'] = performance_notes
        
        # Improvement areas
        print("\n📈 Key Areas for Improvement:")
        print("   What are the main weaknesses that need attention?")
        improvement_areas = input("   > ").strip()
        feedback_data['improvement_areas'] = improvement_areas
        
        # Positive aspects
        print("\n✨ Positive Aspects:")
        print("   What worked well in this episode?")
        positive_aspects = input("   > ").strip()
        feedback_data['positive_aspects'] = positive_aspects
        
        return feedback_data
    
    def _save_feedback(self, feedback_data: Dict[str, Any]):
        """Save feedback to CSV file."""
        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'episode', 'timestamp', 'rating', 'rating_label', 'comments',
                    'suggestions', 'total_reward', 'steps', 'assets_used',
                    'performance_notes', 'improvement_areas', 'positive_aspects'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(feedback_data)
            
            logger.info(f"Feedback saved to {self.csv_file}")
            
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
    
    def _display_feedback_summary(self, feedback_data: Dict[str, Any]):
        """Display a summary of the collected feedback."""
        print("\n" + "="*80)
        print("📄 FEEDBACK SUMMARY")
        print("="*80)
        print(f"Episode: {feedback_data['episode']}")
        print(f"Rating: {feedback_data['rating']}/10 - {feedback_data['rating_label']}")
        print(f"Timestamp: {feedback_data['timestamp']}")
        
        if feedback_data['comments']:
            print(f"Comments: {feedback_data['comments']}")
        
        if feedback_data['suggestions']:
            print(f"Suggestions: {feedback_data['suggestions']}")
        
        print("✅ Feedback successfully recorded!")
        print("="*80)
    
    def interactive_mode(self):
        """Run in interactive mode for continuous feedback collection."""
        print("\n🔄 INTERACTIVE FEEDBACK MODE")
        print("Enter episode data manually or type 'quit' to exit.")
        
        while True:
            try:
                print("\n" + "-" * 50)
                episode_input = input("Episode number (or 'quit'): ").strip()
                
                if episode_input.lower() == 'quit':
                    break
                
                episode = int(episode_input)
                reward = float(input("Total reward: ").strip())
                steps = int(input("Steps taken: ").strip())
                
                assets_input = input("NeRF assets used (comma-separated, or press Enter for none): ").strip()
                assets = [asset.strip() for asset in assets_input.split(',')] if assets_input else []
                
                scene_desc = input("Scene description (optional): ").strip()
                
                episode_data = {
                    'episode': episode,
                    'total_reward': reward,
                    'steps': steps,
                    'assets_used': assets,
                    'scene_description': scene_desc,
                    'performance_metrics': {}
                }
                
                self.collect_feedback(episode_data)
                
            except KeyboardInterrupt:
                break
            except ValueError:
                print("❌ Please enter valid numbers for episode, reward, and steps.")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Exiting interactive mode.")
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load episode data from JSON config file."""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return {}
    
    def display_statistics(self):
        """Display statistics from collected feedback."""
        if not self.csv_file.exists():
            print("No feedback data available.")
            return
        
        try:
            import pandas as pd
            df = pd.read_csv(self.csv_file)
            
            print("\n📊 FEEDBACK STATISTICS")
            print("="*50)
            print(f"Total feedback entries: {len(df)}")
            print(f"Average rating: {df['rating'].mean():.2f}")
            print(f"Rating distribution:")
            
            for rating in range(1, 11):
                count = len(df[df['rating'] == rating])
                percentage = (count / len(df)) * 100 if len(df) > 0 else 0
                bar = "█" * int(percentage / 5)  # Scale bar
                print(f"  {rating:2d}: {count:3d} ({percentage:5.1f}%) {bar}")
            
            print(f"\nMost recent feedback: {df.iloc[-1]['timestamp'] if len(df) > 0 else 'None'}")
            
        except ImportError:
            print("Install pandas for detailed statistics: pip install pandas")
        except Exception as e:
            logger.error(f"Error displaying statistics: {e}")

def main():
    """Main function for CLI tool."""
    parser = argparse.ArgumentParser(
        description="HITL Feedback Collection Tool - Version 5 BETA 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --episode 100 --reward 15.7 --steps 250
  %(prog)s --config episode_data.json
  %(prog)s --interactive
  %(prog)s --stats
        """
    )
    
    parser.add_argument('--episode', type=int, help='Episode number')
    parser.add_argument('--reward', type=float, help='Total reward achieved')
    parser.add_argument('--steps', type=int, help='Number of steps taken')
    parser.add_argument('--assets', nargs='*', help='NeRF assets used')
    parser.add_argument('--scene', help='Scene description')
    parser.add_argument('--config', help='JSON config file with episode data')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--stats', action='store_true', help='Display feedback statistics')
    parser.add_argument('--csv-file', default='hitl_feedback_cli.csv', help='CSV file for feedback storage')
    
    args = parser.parse_args()
    
    # Initialize tool
    tool = HITLCLITool(csv_file=args.csv_file)
    
    # Handle different modes
    if args.stats:
        tool.display_statistics()
        return
    
    if args.interactive:
        tool.interactive_mode()
        return
    
    if args.config:
        episode_data = tool.load_config(args.config)
        if episode_data:
            tool.collect_feedback(episode_data)
        return
    
    if args.episode is not None:
        episode_data = {
            'episode': args.episode,
            'total_reward': args.reward or 0.0,
            'steps': args.steps or 0,
            'assets_used': args.assets or [],
            'scene_description': args.scene or '',
            'performance_metrics': {}
        }
        tool.collect_feedback(episode_data)
        return
    
    # No arguments provided, show help
    parser.print_help()

if __name__ == "__main__":
    main()

