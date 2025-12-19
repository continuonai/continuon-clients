import argparse
import sys
import os
import subprocess
from pathlib import Path
from getpass import getpass

try:
    from fabric import Connection
except ImportError:
    print("Error: fabric package not installed. Run 'pip install fabric'")
    sys.exit(1)

def generate_key_pair(key_path: Path):
    if key_path.exists():
        print(f"ℹ️  Using existing key: {key_path}")
        return

    print(f"🔑 Generating new SSH key pair: {key_path}")
    key_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-f", str(key_path), "-N", ""],
        check=True
    )

def deploy_key(host: str, user: str, password: str, key_path: Path):
    pub_key = key_path.with_suffix(".pub").read_text().strip()
    
    print(f"🚀 Deploying key to {user}@{host}...")
    try:
        # Use fabric with password to deploy key
        c = Connection(host=host, user=user, connect_kwargs={"password": password})
        
        # Create .ssh directory
        c.run("mkdir -p ~/.ssh && chmod 700 ~/.ssh")
        
        # Append key to authorized_keys if not present
        c.run(f"grep -q '{pub_key}' ~/.ssh/authorized_keys || echo '{pub_key}' >> ~/.ssh/authorized_keys")
        c.run("chmod 600 ~/.ssh/authorized_keys")
        print("✅ Key deployed successfully.")
    except Exception as e:
        print(f"❌ Failed to deploy key: {e}")
        sys.exit(1)

def disable_password_auth(host: str, user: str, key_path: Path):
    print("🔒 Disabling password authentication...")
    try:
        # Connect using the KEY this time
        c = Connection(host=host, user=user, connect_kwargs={"key_filename": str(key_path)})
        
        sshd_config = "/etc/ssh/sshd_config"
        backup = f"{sshd_config}.bak"
        
        # Backup config
        c.sudo(f"cp {sshd_config} {backup}")
        
        # Disable PasswordAuthentication
        # We use sed to replace or append.
        c.sudo(f"sed -i 's/^#PasswordAuthentication yes/PasswordAuthentication no/' {sshd_config}")
        c.sudo(f"sed -i 's/^PasswordAuthentication yes/PasswordAuthentication no/' {sshd_config}")
        
        # Ensure ChallengeResponseAuthentication is no
        c.sudo(f"sed -i 's/^#ChallengeResponseAuthentication yes/ChallengeResponseAuthentication no/' {sshd_config}")
        c.sudo(f"sed -i 's/^ChallengeResponseAuthentication yes/ChallengeResponseAuthentication no/' {sshd_config}")

        # Restart SSH
        print("🔄 Restarting SSH service...")
        c.sudo("systemctl restart ssh")
        
        print("✅ Password authentication disabled.")
    except Exception as e:
        print(f"❌ Failed to harden SSH config: {e}")
        print("   Make sure you have sudo privileges and the key is working.")

def main():
    parser = argparse.ArgumentParser(description="Setup SSH keys and security on Pi 5")
    parser.add_argument("host", help="Robot IP address")
    parser.add_argument("--user", default="pi", help="SSH username (default: pi)")
    parser.add_argument("--key-path", default=os.path.expanduser("~/.ssh/continuon_id_ed25519"), help="Path to local private key")
    
    args = parser.parse_args()
    key_path = Path(args.key_path)

    # 1. Generate Key
    generate_key_pair(key_path)

    # 2. Ask for current password
    password = getpass(f"Enter password for {args.user}@{args.host}: ")

    # 3. Deploy Key
    deploy_key(args.host, args.user, password, key_path)

    # 4. Verify connection with key
    print("Testing connection with new key...")
    try:
        c = Connection(host=args.host, user=args.user, connect_kwargs={"key_filename": str(key_path)})
        c.run("echo 'Key authentication working!'", hide=True)
    except Exception as e:
        print(f"❌ Key verification failed: {e}")
        print("   Aborting security hardening.")
        sys.exit(1)

    # 5. Harden (Disable passwords)
    harden = input("Do you want to disable password authentication now? (y/n): ").lower()
    if harden == 'y':
        disable_password_auth(args.host, args.user, key_path)
    else:
        print("ℹ️  Skipping security hardening.")

if __name__ == "__main__":
    main()
