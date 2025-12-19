#!/home/craigm26/Downloads/ContinuonXR/venv/bin/python3
import requests
import json
import time
import sys
from rich.console import Console
from rich.panel import Panel

console = Console()

BASE_URL = "http://127.0.0.1:8081"

def wait_for_server(timeout=60):
    console.print(f"[yellow]Waiting for server at {BASE_URL}...[/yellow]")
    start = time.time()
    while time.time() - start < timeout:
        try:
            requests.get(f"{BASE_URL}/api/ping", timeout=30)
            console.print("[green]Server is up![/green]")
            return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            console.print(f"[dim]Waiting... ({type(e).__name__})[/dim]")
            time.sleep(2)
            continue
    console.print("[bold red]Timeout waiting for server.[/bold red]")
    return False

def run_chat_learn():
    if not wait_for_server():
        return False
        
    console.print("[bold blue]1. Starting Multi-Turn Learning Session (HOPE <-> Gemini)...[/bold blue]")
    
    payload = {
        "turns": 6,
        "model_hint": "hope-v1",
        "delegate_model_hint": "consult:gemini", # Trigger our new logic
        "topic": "system architecture, safety protocols, and continuous learning",
        "session_id": f"gemini_learn_{int(time.time())}"
    }

    try:
        start_time = time.time()
        resp = requests.post(f"{BASE_URL}/api/learning/chat_learn", json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        
        if not data.get("success", False):
            console.print(f"[bold red]Chat Learn Failed:[/bold red] {data}")
            return False
            
        result = data.get("result", {})
        history = result.get("history", [])
        
        console.print(f"[green]Session Complete ({time.time() - start_time:.1f}s)[/green]")
        console.print("\n[bold]Conversation Log:[/bold]")
        
        for msg in history:
            role = msg.get("role", "unknown")
            color = "cyan" if role == "user" else "magenta" # user=Agent Manager (in this context), assistant=Subagent/Gemini
            console.print(Panel(msg.get("content", ""), title=f"[bold {color}]{role.upper()}[/bold {color}]", border_style=color))
            
        return True
    except Exception as e:
        console.print(f"[bold red]Error running Chat Learn:[/bold red] {e}")
        return False

def run_training():
    console.print("\n[bold blue]2. Kicking off Formal Training Run (Manual JAX)...[/bold blue]")
    
    # Use Manual Training to digest the episodes just created
    payload = {
        "max_steps": 20,
        "batch_size": 4,
        "use_synthetic": False, # Use real RLDS data!
        "learning_rate": 1e-4
    }
    
    try:
        resp = requests.post(f"{BASE_URL}/api/training/manual", json=payload, timeout=60)
         # Note: Manual endpoint might return directly or async. 
         # BrainService.RunManualTraining is async but wrapper waits? 
         # Let's check status.
        
        if resp.status_code == 200:
            console.print("[green]Training Run Initiated/Complete.[/green]")
            console.print(resp.json())
        else:
            console.print(f"[red]Training Request Failed: {resp.status_code} {resp.text}[/red]")
            
    except Exception as e:
        console.print(f"[bold red]Error running training:[/bold red] {e}")

if __name__ == "__main__":
    console.print(Panel("[bold yellow]HOPE Agent Manager - Learning Session[/bold yellow]", subtitle="Powered by Gemini"))
    
    success = run_chat_learn()
    
    if success:
        run_training()
    else:
        console.print("[red]Skipping training due to learning session failure.[/red]")
