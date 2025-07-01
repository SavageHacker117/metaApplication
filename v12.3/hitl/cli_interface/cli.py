"""
Human-in-the-Loop CLI Interface

This module provides a command-line interface for human feedback collection
in the RL training process.
"""

import click
import json
import csv
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    """Human-in-the-Loop CLI for RL training feedback."""
    ctx.ensure_object(dict)
    
    # Setup logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    ctx.obj['verbose'] = verbose


@cli.command()
@click.option('--episode', '-e', type=int, required=True, help='Episode number')
@click.option('--reward', '-r', type=float, required=True, help='Episode reward')
@click.option('--steps', '-s', type=int, required=True, help='Episode steps')
@click.option('--output', '-o', default='feedback.csv', help='Output CSV file')
@click.pass_context
def rate_episode(ctx, episode, reward, steps, output):
    """Rate a training episode interactively."""
    click.echo(f"Rating Episode {episode}")
    click.echo(f"Reward: {reward:.2f}")
    click.echo(f"Steps: {steps}")
    click.echo("-" * 40)
    
    # Get rating
    rating = click.prompt(
        'Rate this episode (1-10, where 10 is excellent)',
        type=click.IntRange(1, 10)
    )
    
    # Get detailed feedback
    strategy_rating = click.prompt(
        'Rate the strategy (1-10)',
        type=click.IntRange(1, 10)
    )
    
    efficiency_rating = click.prompt(
        'Rate the efficiency (1-10)',
        type=click.IntRange(1, 10)
    )
    
    comments = click.prompt(
        'Additional comments (optional)',
        default='',
        show_default=False
    )
    
    # Collect feedback data
    feedback_data = {
        'timestamp': datetime.now().isoformat(),
        'episode': episode,
        'reward': reward,
        'steps': steps,
        'overall_rating': rating,
        'strategy_rating': strategy_rating,
        'efficiency_rating': efficiency_rating,
        'comments': comments
    }
    
    # Save to CSV
    save_feedback_csv(feedback_data, output)
    
    click.echo(f"✓ Feedback saved to {output}")
    
    if ctx.obj['verbose']:
        click.echo(f"Feedback data: {json.dumps(feedback_data, indent=2)}")


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file')
@click.option('--episodes', '-e', type=int, default=5, help='Number of episodes to rate')
@click.option('--output', '-o', default='batch_feedback.csv', help='Output CSV file')
def batch_rate(config, episodes, output):
    """Rate multiple episodes in batch mode."""
    if config:
        with open(config, 'r') as f:
            episode_data = json.load(f)
    else:
        # Generate sample episode data
        episode_data = generate_sample_episodes(episodes)
    
    click.echo(f"Batch rating {len(episode_data)} episodes")
    click.echo("=" * 50)
    
    all_feedback = []
    
    for i, episode_info in enumerate(episode_data):
        click.echo(f"\nEpisode {i+1}/{len(episode_data)}")
        click.echo(f"Episode ID: {episode_info.get('episode', i)}")
        click.echo(f"Reward: {episode_info.get('reward', 0):.2f}")
        click.echo(f"Steps: {episode_info.get('steps', 0)}")
        
        if 'description' in episode_info:
            click.echo(f"Description: {episode_info['description']}")
        
        click.echo("-" * 30)
        
        # Quick rating
        rating = click.prompt(
            'Quick rating (1-10) or "s" to skip',
            type=str
        )
        
        if rating.lower() == 's':
            continue
        
        try:
            rating = int(rating)
            if not 1 <= rating <= 10:
                raise ValueError()
        except ValueError:
            click.echo("Invalid rating, skipping episode")
            continue
        
        feedback_data = {
            'timestamp': datetime.now().isoformat(),
            'episode': episode_info.get('episode', i),
            'reward': episode_info.get('reward', 0),
            'steps': episode_info.get('steps', 0),
            'overall_rating': rating,
            'batch_mode': True
        }
        
        all_feedback.append(feedback_data)
    
    # Save all feedback
    for feedback in all_feedback:
        save_feedback_csv(feedback, output)
    
    click.echo(f"\n✓ Saved {len(all_feedback)} feedback entries to {output}")


@cli.command()
@click.option('--input', '-i', default='feedback.csv', help='Input CSV file')
@click.option('--format', '-f', type=click.Choice(['summary', 'detailed', 'json']), 
              default='summary', help='Report format')
def analyze(input, format):
    """Analyze collected feedback data."""
    input_path = Path(input)
    
    if not input_path.exists():
        click.echo(f"Error: Feedback file {input} not found", err=True)
        return
    
    # Load feedback data
    feedback_data = load_feedback_csv(input)
    
    if not feedback_data:
        click.echo("No feedback data found")
        return
    
    # Generate analysis
    analysis = analyze_feedback(feedback_data)
    
    if format == 'summary':
        display_summary(analysis)
    elif format == 'detailed':
        display_detailed_analysis(analysis)
    elif format == 'json':
        click.echo(json.dumps(analysis, indent=2))


@cli.command()
@click.option('--input', '-i', default='feedback.csv', help='Input CSV file')
@click.option('--output', '-o', help='Output file (optional)')
def export(input, output):
    """Export feedback data in various formats."""
    input_path = Path(input)
    
    if not input_path.exists():
        click.echo(f"Error: Feedback file {input} not found", err=True)
        return
    
    feedback_data = load_feedback_csv(input)
    
    if output:
        output_path = Path(output)
        if output_path.suffix == '.json':
            with open(output_path, 'w') as f:
                json.dump(feedback_data, f, indent=2)
            click.echo(f"✓ Exported to JSON: {output_path}")
        else:
            click.echo("Unsupported output format")
    else:
        # Print to stdout
        click.echo(json.dumps(feedback_data, indent=2))


def save_feedback_csv(feedback_data: Dict[str, Any], filepath: str):
    """Save feedback data to CSV file."""
    filepath = Path(filepath)
    
    # Check if file exists to determine if we need headers
    file_exists = filepath.exists()
    
    with open(filepath, 'a', newline='') as csvfile:
        fieldnames = [
            'timestamp', 'episode', 'reward', 'steps', 
            'overall_rating', 'strategy_rating', 'efficiency_rating', 
            'comments', 'batch_mode'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(feedback_data)


def load_feedback_csv(filepath: str) -> List[Dict[str, Any]]:
    """Load feedback data from CSV file."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        return []
    
    feedback_data = []
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
    
    return feedback_data


def analyze_feedback(feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze feedback data and generate statistics."""
    if not feedback_data:
        return {}
    
    # Extract ratings
    overall_ratings = [f.get('overall_rating') for f in feedback_data if f.get('overall_rating')]
    strategy_ratings = [f.get('strategy_rating') for f in feedback_data if f.get('strategy_rating')]
    efficiency_ratings = [f.get('efficiency_rating') for f in feedback_data if f.get('efficiency_rating')]
    
    rewards = [f.get('reward') for f in feedback_data if f.get('reward') is not None]
    steps = [f.get('steps') for f in feedback_data if f.get('steps')]
    
    analysis = {
        'total_episodes': len(feedback_data),
        'rating_stats': {},
        'performance_stats': {}
    }
    
    # Rating statistics
    if overall_ratings:
        analysis['rating_stats']['overall'] = {
            'mean': sum(overall_ratings) / len(overall_ratings),
            'min': min(overall_ratings),
            'max': max(overall_ratings),
            'count': len(overall_ratings)
        }
    
    if strategy_ratings:
        analysis['rating_stats']['strategy'] = {
            'mean': sum(strategy_ratings) / len(strategy_ratings),
            'min': min(strategy_ratings),
            'max': max(strategy_ratings),
            'count': len(strategy_ratings)
        }
    
    if efficiency_ratings:
        analysis['rating_stats']['efficiency'] = {
            'mean': sum(efficiency_ratings) / len(efficiency_ratings),
            'min': min(efficiency_ratings),
            'max': max(efficiency_ratings),
            'count': len(efficiency_ratings)
        }
    
    # Performance statistics
    if rewards:
        analysis['performance_stats']['reward'] = {
            'mean': sum(rewards) / len(rewards),
            'min': min(rewards),
            'max': max(rewards),
            'count': len(rewards)
        }
    
    if steps:
        analysis['performance_stats']['steps'] = {
            'mean': sum(steps) / len(steps),
            'min': min(steps),
            'max': max(steps),
            'count': len(steps)
        }
    
    return analysis


def display_summary(analysis: Dict[str, Any]):
    """Display summary analysis."""
    click.echo("Feedback Analysis Summary")
    click.echo("=" * 30)
    
    click.echo(f"Total Episodes: {analysis.get('total_episodes', 0)}")
    
    if 'rating_stats' in analysis:
        click.echo("\nRating Statistics:")
        for rating_type, stats in analysis['rating_stats'].items():
            click.echo(f"  {rating_type.capitalize()}: {stats['mean']:.2f} "
                      f"(min: {stats['min']}, max: {stats['max']}, n: {stats['count']})")
    
    if 'performance_stats' in analysis:
        click.echo("\nPerformance Statistics:")
        for perf_type, stats in analysis['performance_stats'].items():
            click.echo(f"  {perf_type.capitalize()}: {stats['mean']:.2f} "
                      f"(min: {stats['min']}, max: {stats['max']}, n: {stats['count']})")


def display_detailed_analysis(analysis: Dict[str, Any]):
    """Display detailed analysis."""
    display_summary(analysis)
    
    # Add more detailed information here
    click.echo("\nDetailed Analysis:")
    click.echo("(Additional detailed metrics would go here)")


def generate_sample_episodes(count: int) -> List[Dict[str, Any]]:
    """Generate sample episode data for testing."""
    import random
    
    episodes = []
    for i in range(count):
        episodes.append({
            'episode': i + 1,
            'reward': random.uniform(-10, 50),
            'steps': random.randint(50, 500),
            'description': f"Sample episode {i + 1} for testing"
        })
    
    return episodes


if __name__ == '__main__':
    cli()

