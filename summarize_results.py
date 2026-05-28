#!/usr/bin/env python3
"""
Generate aggregate summaries for each result type (folder).
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def load_results(json_file: str) -> dict:
    """Load evaluation results from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def aggregate_by_type(all_results: dict) -> dict:
    """
    Aggregate results by folder type.
    
    Returns:
        Dictionary with 4 types: results, results_0_norev, results_6, results_6_norev
    """
    type_results = {
        'results': [],
        'results_0_norev': [],
        'results_6': [],
        'results_6_norev': []
    }
    
    for filepath, results in all_results.items():
        path = Path(filepath)
        folder_name = path.parent.name
        
        if folder_name in type_results:
            metrics = results['aggregated_metrics']
            filename = path.name
            
            # Extract system name
            if 'cand_a' in filename:
                system = 'cand_a'
            elif 'cand_b' in filename:
                system = 'cand_b'
            elif 'cand_c' in filename:
                system = 'cand_c'
            elif 'my_system' in filename:
                system = 'my_system'
            else:
                system = 'unknown'
            
            type_results[folder_name].append({
                'system': system,
                'file': filename,
                'num_evaluated': results['num_evaluated'],
                'chrfpp': metrics['average_chrfpp'],
                'term_accuracy': metrics['overall_terminology_accuracy'],
                'total_terms': metrics['total_proper_terms'],
                'found_terms': metrics['found_proper_terms']
            })
    
    return type_results


def calculate_type_summary(type_data: list) -> dict:
    """Calculate aggregate summary for a result type."""
    if not type_data:
        return None
    
    # Aggregate across all systems in this type
    total_evaluated = sum(item['num_evaluated'] for item in type_data)
    total_terms = sum(item['total_terms'] for item in type_data)
    total_found = sum(item['found_terms'] for item in type_data)
    
    # Weighted average chrF++ (by number of evaluated entries)
    weighted_chrfpp = sum(
        item['chrfpp'] * item['num_evaluated'] 
        for item in type_data
    ) / total_evaluated if total_evaluated > 0 else 0.0
    
    # Overall terminology accuracy
    overall_term_accuracy = total_found / total_terms if total_terms > 0 else 0.0
    
    # Per-system statistics
    systems = {}
    for item in type_data:
        system = item['system']
        if system not in systems:
            systems[system] = {
                'chrfpp': [],
                'term_accuracy': [],
                'num_evaluated': 0,
                'total_terms': 0,
                'found_terms': 0
            }
        
        systems[system]['chrfpp'].append(item['chrfpp'])
        systems[system]['term_accuracy'].append(item['term_accuracy'])
        systems[system]['num_evaluated'] += item['num_evaluated']
        systems[system]['total_terms'] += item['total_terms']
        systems[system]['found_terms'] += item['found_terms']
    
    # Calculate averages for each system
    system_stats = {}
    for system, data in systems.items():
        system_stats[system] = {
            'avg_chrfpp': sum(data['chrfpp']) / len(data['chrfpp']),
            'avg_term_accuracy': data['found_terms'] / data['total_terms'] if data['total_terms'] > 0 else 0.0,
            'num_evaluated': data['num_evaluated'],
            'total_terms': data['total_terms'],
            'found_terms': data['found_terms']
        }
    
    return {
        'total_evaluated': total_evaluated,
        'weighted_avg_chrfpp': weighted_chrfpp,
        'overall_term_accuracy': overall_term_accuracy,
        'total_terms': total_terms,
        'total_found_terms': total_found,
        'num_systems': len(type_data),
        'systems': system_stats
    }


def format_type_summary(type_name: str, summary: dict) -> str:
    """Format a summary for one type."""
    lines = []
    lines.append("=" * 100)
    lines.append(f"AGGREGATE SUMMARY: {type_name.upper()}")
    lines.append("=" * 100)
    
    lines.append(f"\nOverall Statistics:")
    lines.append(f"  Total Evaluated Entries: {summary['total_evaluated']:,}")
    lines.append(f"  Weighted Average chrF++: {summary['weighted_avg_chrfpp']:.2f}")
    lines.append(f"  Overall Terminology Accuracy: {summary['overall_term_accuracy']:.2%}")
    lines.append(f"  Found Terms: {summary['total_found_terms']:,}/{summary['total_terms']:,}")
    lines.append(f"  Number of Systems: {summary['num_systems']}")
    
    lines.append(f"\nPer-System Breakdown:")
    lines.append("-" * 100)
    
    # Sort systems by chrF++ score
    sorted_systems = sorted(
        summary['systems'].items(),
        key=lambda x: x[1]['avg_chrfpp'],
        reverse=True
    )
    
    for system, stats in sorted_systems:
        lines.append(f"\n  {system.upper()}:")
        lines.append(f"    Average chrF++: {stats['avg_chrfpp']:.2f}")
        lines.append(f"    Terminology Accuracy: {stats['avg_term_accuracy']:.2%}")
        lines.append(f"    Evaluated Entries: {stats['num_evaluated']:,}")
        lines.append(f"    Found Terms: {stats['found_terms']:,}/{stats['total_terms']:,}")
    
    lines.append("\n" + "=" * 100)
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        json_file = 'evaluation_results.json'
    else:
        json_file = sys.argv[1]
    
    print(f"Loading results from {json_file}...", file=sys.stderr)
    data = load_results(json_file)
    all_results = data['all_results']
    
    print("Aggregating by type...", file=sys.stderr)
    type_results = aggregate_by_type(all_results)
    
    # Generate summaries for each type
    summaries = {}
    for type_name, type_data in type_results.items():
        summary = calculate_type_summary(type_data)
        if summary:
            summaries[type_name] = summary
    
    # Print all summaries
    type_order = ['results', 'results_0_norev', 'results_6', 'results_6_norev']
    
    for type_name in type_order:
        if type_name in summaries:
            print(format_type_summary(type_name, summaries[type_name]))
            print()
    
    # Optional: Save to JSON
    if len(sys.argv) > 2 and sys.argv[2] == '--json':
        output = {
            'summaries': {
                type_name: summaries[type_name]
                for type_name in type_order
                if type_name in summaries
            }
        }
        output_file = json_file.replace('.json', '_summaries.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        print(f"Summaries saved to {output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()






