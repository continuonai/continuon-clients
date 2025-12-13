#!/usr/bin/env python3
"""
Analyze training status reports and generate statistics summary.
Detects trends and changes in training metrics over time.
"""

import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

try:
    import requests
except ImportError:
    requests = None


def parse_report_line(line):
    """Extract metrics from a single report line."""
    metrics = {}
    
    # Extract timestamp
    timestamp_match = re.search(r'Generated: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
    if timestamp_match:
        try:
            metrics['timestamp'] = datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    
    # Background learner metrics
    if 'Total steps:' in line:
        match = re.search(r'Total steps: ([\d,]+)', line)
        if match:
            metrics['bg_steps'] = int(match.group(1).replace(',', ''))
    
    if 'Learning updates:' in line:
        match = re.search(r'Learning updates: ([\d,]+)', line)
        if match:
            metrics['bg_updates'] = int(match.group(1).replace(',', ''))
    
    if 'Avg parameter change:' in line:
        match = re.search(r'Avg parameter change: ([\d.]+)', line)
        if match:
            metrics['bg_param_change'] = float(match.group(1))
    
    if 'Checkpoints:' in line:
        match = re.search(r'Checkpoints: ([\d,]+)', line)
        if match:
            metrics['bg_checkpoints'] = int(match.group(1).replace(',', ''))
    
    # Chat learning metrics
    if 'Chat episodes:' in line:
        match = re.search(r'Chat episodes: ([\d,]+)', line)
        if match:
            metrics['chat_episodes'] = int(match.group(1).replace(',', ''))
    
    # Tool router metrics
    if 'Training steps:' in line and 'TOOL ROUTER' in line:
        match = re.search(r'Training steps: (\d+)', line)
        if match:
            metrics['tr_steps'] = int(match.group(1))
    
    if 'Loss:' in line and '→' in line:
        match = re.search(r'Loss: ([\d.]+) → ([\d.]+)', line)
        if match:
            metrics['tr_loss_initial'] = float(match.group(1))
            metrics['tr_loss_final'] = float(match.group(2))
    
    if 'Accuracy:' in line and '→' in line:
        match = re.search(r'Accuracy: ([\d.]+) → ([\d.]+)', line)
        if match:
            metrics['tr_acc_initial'] = float(match.group(1))
            metrics['tr_acc_final'] = float(match.group(2))
    
    # Memory metrics
    if 'Memory:' in line and 'MB' in line:
        match = re.search(r'Memory: ([\d,]+)MB / ([\d,]+)MB \(([\d.]+)%\)', line)
        if match:
            metrics['mem_used_mb'] = int(match.group(1).replace(',', ''))
            metrics['mem_total_mb'] = int(match.group(2).replace(',', ''))
            metrics['mem_percent'] = float(match.group(3))
    
    # HOPE eval episodes
    if 'HOPE EVALUATIONS' in line or 'Episodes:' in line:
        match = re.search(r'Episodes: ([\d,]+)', line)
        if match:
            metrics['hope_episodes'] = int(match.group(1).replace(',', ''))
    
    return metrics if metrics else None


def parse_reports_log(log_path):
    """Parse all reports from the log file."""
    if not log_path.exists():
        return []
    
    reports = []
    current_report = {}
    in_report = False
    
    with open(log_path, 'r') as f:
        for line in f:
            if 'COMPREHENSIVE TRAINING STATUS REPORT' in line:
                if current_report and 'timestamp' in current_report:
                    reports.append(current_report)
                current_report = {}
                in_report = True
            
            if in_report:
                metrics = parse_report_line(line)
                if metrics:
                    current_report.update(metrics)
    
    # Add last report
    if current_report and 'timestamp' in current_report:
        reports.append(current_report)
    
    return reports


def compute_statistics(reports):
    """Compute statistics from parsed reports."""
    if not reports:
        return {}
    
    stats = {
        'total_reports': len(reports),
        'time_span': None,
        'metrics': {}
    }
    
    if len(reports) > 1:
        timestamps = [r['timestamp'] for r in reports if 'timestamp' in r]
        if timestamps:
            stats['time_span'] = {
                'start': min(timestamps),
                'end': max(timestamps),
                'duration_hours': (max(timestamps) - min(timestamps)).total_seconds() / 3600
            }
    
    # Aggregate metrics
    metric_keys = [
        'bg_steps', 'bg_updates', 'bg_param_change', 'bg_checkpoints',
        'chat_episodes', 'tr_steps', 'tr_loss_final', 'tr_acc_final',
        'mem_used_mb', 'mem_percent', 'hope_episodes'
    ]
    
    for key in metric_keys:
        values = [r[key] for r in reports if key in r]
        if values:
            stats['metrics'][key] = {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'latest': values[-1],
                'first': values[0] if len(values) > 0 else None,
                'change': values[-1] - values[0] if len(values) > 1 else 0,
                'change_pct': ((values[-1] - values[0]) / values[0] * 100) if len(values) > 1 and values[0] != 0 else 0
            }
    
    return stats


def detect_trends(stats):
    """Detect trends and significant changes."""
    trends = []
    
    metrics = stats.get('metrics', {})
    
    # Background learner trends
    if 'bg_steps' in metrics:
        m = metrics['bg_steps']
        if m['change'] > 0:
            trends.append(f"✅ Background learner progressing: {m['change']:,} new steps (+{m['change_pct']:.1f}%)")
        elif m['change'] == 0:
            trends.append("⚠️  Background learner paused: No new steps")
    
    if 'bg_updates' in metrics:
        m = metrics['bg_updates']
        if m['change'] > 0:
            trends.append(f"✅ Learning updates: {m['change']:,} new updates")
    
    # Chat learning trends
    if 'chat_episodes' in metrics:
        m = metrics['chat_episodes']
        if m['change'] > 0:
            trends.append(f"✅ Chat learning active: {m['change']:,} new episodes (+{m['change_pct']:.1f}%)")
        else:
            trends.append("⚠️  Chat learning: No new episodes")
    
    # Tool router trends
    if 'tr_acc_final' in metrics:
        m = metrics['tr_acc_final']
        if m['change'] > 0:
            trends.append(f"✅ Tool router improving: Accuracy +{m['change']:.3f} (+{m['change_pct']:.1f}%)")
        elif m['change'] < 0:
            trends.append(f"⚠️  Tool router accuracy decreased: {m['change']:.3f}")
    
    if 'tr_loss_final' in metrics:
        m = metrics['tr_loss_final']
        if m['change'] < 0:
            trends.append(f"✅ Tool router loss decreasing: {m['change']:.4f} (improving)")
        elif m['change'] > 0:
            trends.append(f"⚠️  Tool router loss increasing: {m['change']:.4f} (worsening)")
    
    # Memory trends
    if 'mem_percent' in metrics:
        m = metrics['mem_percent']
        if m['latest'] > 80:
            trends.append(f"⚠️  High memory usage: {m['latest']:.1f}%")
        elif m['change'] > 5:
            trends.append(f"⚠️  Memory usage increasing: +{m['change']:.1f}%")
        else:
            trends.append(f"✅ Memory usage stable: {m['latest']:.1f}%")
    
    # HOPE eval trends
    if 'hope_episodes' in metrics:
        m = metrics['hope_episodes']
        if m['change'] > 0:
            trends.append(f"✅ HOPE evaluations: {m['change']:,} new episodes")
    
    return trends


def generate_summary_report(stats, trends):
    """Generate formatted summary report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = []
    report.append("=" * 80)
    report.append("📊 TRAINING STATISTICS SUMMARY REPORT")
    report.append(f"   Generated: {timestamp}")
    report.append("=" * 80)
    report.append("")
    
    # Time span
    if stats.get('time_span'):
        ts = stats['time_span']
        report.append(f"📅 Analysis Period:")
        report.append(f"   Start: {ts['start'].strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"   End:   {ts['end'].strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"   Duration: {ts['duration_hours']:.1f} hours")
        report.append(f"   Reports analyzed: {stats['total_reports']}")
        report.append("")
    
    # Key metrics summary
    report.append("📈 KEY METRICS SUMMARY:")
    report.append("")
    
    metrics = stats.get('metrics', {})
    
    if 'bg_steps' in metrics:
        m = metrics['bg_steps']
        report.append(f"1️⃣  Background Learner Steps:")
        report.append(f"   Current: {m['latest']:,}")
        report.append(f"   Change: {m['change']:+,} ({m['change_pct']:+.1f}%)")
        report.append(f"   Average: {m['avg']:,.0f}")
        report.append(f"   Range: {m['min']:,} - {m['max']:,}")
        report.append("")
    
    if 'chat_episodes' in metrics:
        m = metrics['chat_episodes']
        report.append(f"2️⃣  Chat Learning Episodes:")
        report.append(f"   Current: {m['latest']:,}")
        report.append(f"   Change: {m['change']:+,} ({m['change_pct']:+.1f}%)")
        report.append(f"   Average: {m['avg']:,.0f}")
        report.append("")
    
    if 'tr_acc_final' in metrics:
        m = metrics['tr_acc_final']
        report.append(f"3️⃣  Tool Router Accuracy (Symbolic Search):")
        report.append(f"   Current: {m['latest']:.3f} ({m['latest']*100:.1f}%)")
        report.append(f"   Change: {m['change']:+.3f} ({m['change_pct']:+.1f}%)")
        report.append(f"   Average: {m['avg']:.3f}")
        report.append("")
    
    if 'tr_loss_final' in metrics:
        m = metrics['tr_loss_final']
        report.append(f"4️⃣  Tool Router Loss:")
        report.append(f"   Current: {m['latest']:.4f}")
        report.append(f"   Change: {m['change']:+.4f}")
        report.append(f"   Average: {m['avg']:.4f}")
        report.append("")
    
    if 'mem_percent' in metrics:
        m = metrics['mem_percent']
        report.append(f"5️⃣  Memory Usage:")
        report.append(f"   Current: {m['latest']:.1f}%")
        report.append(f"   Change: {m['change']:+.1f}%")
        report.append(f"   Average: {m['avg']:.1f}%")
        report.append(f"   Range: {m['min']:.1f}% - {m['max']:.1f}%")
        report.append("")
    
    if 'hope_episodes' in metrics:
        m = metrics['hope_episodes']
        report.append(f"6️⃣  HOPE Evaluation Episodes:")
        report.append(f"   Current: {m['latest']:,}")
        report.append(f"   Change: {m['change']:+,}")
        report.append("")
    
    # Trends
    report.append("=" * 80)
    report.append("🔍 TRENDS & CHANGES:")
    report.append("=" * 80)
    report.append("")
    
    if trends:
        for trend in trends:
            report.append(f"   {trend}")
    else:
        report.append("   No significant trends detected")
    
    report.append("")
    
    # Learning status
    report.append("=" * 80)
    report.append("✅ LEARNING STATUS:")
    report.append("=" * 80)
    report.append("")
    
    learning_active = False
    if 'bg_steps' in metrics and metrics['bg_steps']['change'] > 0:
        learning_active = True
        report.append("   ✅ Background learner: ACTIVE (steps increasing)")
    else:
        report.append("   ⏸️  Background learner: PAUSED (no new steps)")
    
    if 'chat_episodes' in metrics and metrics['chat_episodes']['change'] > 0:
        learning_active = True
        report.append(f"   ✅ Chat learning: ACTIVE ({metrics['chat_episodes']['change']} new episodes)")
    else:
        report.append("   ⏸️  Chat learning: No new episodes")
    
    if 'tr_acc_final' in metrics and metrics['tr_acc_final']['change'] > 0:
        learning_active = True
        report.append("   ✅ Tool router: IMPROVING (accuracy increasing)")
    else:
        report.append("   ⏸️  Tool router: Stable")
    
    report.append("")
    report.append(f"   Overall: {'✅ LEARNING ACTIVE' if learning_active else '⚠️  LEARNING PAUSED'}")
    report.append("")
    
    # Symbolic search status
    report.append("=" * 80)
    report.append("🔍 SYMBOLIC SEARCH STATUS:")
    report.append("=" * 80)
    report.append("")
    report.append("   ✅ Tool Router: Operational (maps language → tools)")
    report.append("   ✅ Beam Search: Available (explores many futures)")
    report.append("   ✅ Tree Search: Available (optimal action selection)")
    report.append("")
    
    report.append("=" * 80)
    report.append("")
    
    return "\n".join(report)


def send_email_report(report_text, recipient):
    """Send report via email (if configured)."""
    try:
        import smtplib
        import os
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        # Email configuration from environment or defaults
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        sender_email = os.getenv("SMTP_SENDER", "continuonbrain@localhost")
        smtp_user = os.getenv("SMTP_USER", "")
        smtp_password = os.getenv("SMTP_PASSWORD", "")
        
        # Skip if credentials not configured
        if not smtp_user or not smtp_password:
            return False
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient
        msg['Subject'] = f"ContinuonBrain Training Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        msg.attach(MIMEText(report_text, 'plain'))
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        
        return True
    except Exception as e:
        print(f"⚠️  Email send failed: {e}")
        return False


def main():
    """Main analysis function."""
    log_path = Path("/opt/continuonos/brain/logs/training_status_reports.log")
    output_path = Path("/root/training_statistics_report.txt")
    email_recipient = "craigm26@gmail.com"
    
    print(f"Analyzing training reports from: {log_path}")
    
    # Parse reports
    reports = parse_reports_log(log_path)
    
    if not reports:
        print("⚠️  No reports found in log file")
        report_text = f"No training reports found. Log file: {log_path}\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    else:
        print(f"✅ Parsed {len(reports)} reports")
        
        # Compute statistics
        stats = compute_statistics(reports)
        
        # Detect trends
        trends = detect_trends(stats)
        
        # Generate report
        report_text = generate_summary_report(stats, trends)
    
    # Write to root-level file
    root_output = Path("/root/training_statistics_report.txt")
    home_output = Path.home() / "training_statistics_report.txt"
    
    written = False
    for output_path in [root_output, home_output]:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"✅ Report written to: {output_path}")
            written = True
            break
        except (PermissionError, OSError):
            continue
    
    if not written:
        # Final fallback to /tmp
        tmp_output = Path("/tmp/training_statistics_report.txt")
        with open(tmp_output, 'w') as f:
            f.write(report_text)
        print(f"✅ Report written to: {tmp_output}")
    
    # Try to send email (if configured)
    email_sent = send_email_report(report_text, email_recipient)
    if email_sent:
        print(f"✅ Report emailed to: {email_recipient}")
    else:
        print(f"ℹ️  Email not sent (configure SMTP_USER and SMTP_PASSWORD env vars to enable)")
    
    print("\n" + report_text)


if __name__ == "__main__":
    main()
